package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/go-redis/redis/v8"
	"github.com/sirupsen/logrus"
)

type DistributedMaster struct {
	port            int
	redisClient     *redis.Client
	resultHashKey   string
	taskDataHashKey string
	logger          *logrus.Logger
	taskRestCtx     context.Context
	taskRestCancel  context.CancelFunc
	wg              sync.WaitGroup
}

func NewDistributedMaster(port int) *DistributedMaster {
	// 初始化日志
	logger := logrus.New()
	logDir := filepath.Join("..", "..", "logs", "Distribution_logs")
	if err := os.MkdirAll(logDir, 0755); err != nil {
		log.Fatalf("Failed to create log directory: %v", err)
	}
	logFile := filepath.Join(logDir, "master.log")
	file, err := os.OpenFile(logFile, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
	if err != nil {
		log.Fatalf("Failed to open log file: %v", err)
	}
	logger.SetOutput(file)
	logger.SetLevel(logrus.InfoLevel)

	// 初始化Redis
	rdb := redis.NewClient(&redis.Options{
		Addr:     "localhost:6379",
		PoolSize: 256,
	})

	// 清理现有数据
	ctx := context.Background()
	rdb.Del(ctx, "distributed_result", "task_data_hash", "cpu", "gpu")

	// 创建上下文用于后台任务
	taskCtx, cancel := context.WithCancel(context.Background())

	return &DistributedMaster{
		port:            port,
		redisClient:     rdb,
		resultHashKey:   "distributed_result",
		taskDataHashKey: "task_data_hash",
		logger:          logger,
		taskRestCtx:     taskCtx,
		taskRestCancel:  cancel,
	}
}

func (dm *DistributedMaster) Run() {
	router := gin.Default()
	router.Use(dm.checkVerification())

	router.POST("/get_task", dm.getTask)
	router.POST("/get_result", dm.getResult)
	router.POST("/clear_record", dm.clearRecord)
	router.POST("/report_result", dm.reportResult)
	router.POST("/create_task", dm.createTask)

	// 启动后台任务
	dm.wg.Add(1)
	go dm.showTaskRest()

	// 启动HTTP服务器
	srv := &http.Server{
		Addr:    fmt.Sprintf(":%d", dm.port),
		Handler: router,
	}

	go func() {
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			dm.logger.Fatalf("Server error: %v", err)
		}
	}()

	// 等待中断信号
	// 	quit := make(chan os.Signal, 1)
	// 	signal.Notify(quit, os.Interrupt)
	// 	<-quit
	//
	// 	// 优雅关闭
	// 	dm.taskRestCancel()
	// 	dm.wg.Wait()
	//
	// 	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	// 	defer cancel()
	// 	if err := srv.Shutdown(ctx); err != nil {
	// 		dm.logger.Fatal("Server shutdown error:", err)
	// 	}
}

func (dm *DistributedMaster) checkVerification() gin.HandlerFunc {
	return func(c *gin.Context) {
		checkVal := c.PostForm("check_val")
		if checkVal != "81600a92e8416bba7d9fada48e9402a4" {
			c.AbortWithStatusJSON(http.StatusForbidden, gin.H{
				"success": false,
				"msg":     "ERROR",
			})
			return
		}
		c.Next()
	}
}

func (dm *DistributedMaster) getTask(c *gin.Context) {
	taskType := c.PostForm("task_type")
	if taskType != "cpu" && taskType != "gpu" {
		c.JSON(http.StatusInternalServerError, gin.H{
			"success": false,
			"msg":     "TASK Type ERROR",
		})
		return
	}

	maxCost, err := strconv.ParseFloat(c.PostForm("max_cost"), 64)
	if err != nil {
		dm.logger.Error("Invalid max_cost:", err)
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid max_cost"})
		return
	}

	var taskID string
	for {
		// ZRANGEBYSCORE with LIMIT 0 1
		cmd := dm.redisClient.ZRangeByScore(c.Request.Context(), taskType, &redis.ZRangeBy{
			Min:    "0",
			Max:    strconv.FormatFloat(maxCost, 'f', -1, 64),
			Offset: 0,
			Count:  1,
		})

		members, err := cmd.Result()
		if err != nil {
			dm.logger.Error("Redis error:", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		if len(members) == 0 {
			c.JSON(http.StatusInternalServerError, gin.H{
				"success": false,
				"msg":     "There is no Task can RUN",
			})
			return
		}

		// 使用事务确保原子性
		txPipe := dm.redisClient.TxPipeline()
		remCmd := txPipe.ZRem(c.Request.Context(), taskType, members)
		_, err = txPipe.Exec(c.Request.Context())
		if err != nil {
			dm.logger.Error("Transaction failed:", err, "Retry")
			continue
		}

		// 确保唯一取到taskID
		if remCmd.Val() > 0 {
			taskID = members[0]
			break
		}
	}

	//取出对应taskID的taskData
	result, err := dm.redisClient.HGet(c.Request.Context(), dm.taskDataHashKey, taskID).Result()
	if err == redis.Nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"success": false,
			"msg":     "Task LOSS",
		})
		return
	} else if err != nil {
		dm.logger.Error("Redis error:", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	// 删除taskID的taskData记录
	if _, err := dm.redisClient.HDel(c.Request.Context(), dm.taskDataHashKey, taskID).Result(); err != nil {
		dm.logger.Error("Redis error:", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.Data(http.StatusOK, "application/octet-stream", []byte(result))
}

func (dm *DistributedMaster) clearRecord(c *gin.Context) {
	taskID := c.PostForm("task_id")
	if _, err := dm.redisClient.HDel(c.Request.Context(), dm.resultHashKey, taskID).Result(); err != nil {
		dm.logger.Error("Redis error:", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"success": true})
}

func (dm *DistributedMaster) getResult(c *gin.Context) {
	taskID := c.PostForm("task_id")
	result, err := dm.redisClient.HGet(c.Request.Context(), dm.resultHashKey, taskID).Result()
	if err == redis.Nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"success": false,
			"msg":     "Task haven't FINISHED",
		})
		return
	} else if err != nil {
		dm.logger.Error("Redis error:", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.Data(http.StatusOK, "application/octet-stream", []byte(result))
}

func (dm *DistributedMaster) reportResult(c *gin.Context) {
	taskID := c.PostForm("task_id")
	file, err := c.FormFile("result")
	if err != nil {
		dm.logger.Error("File error:", err)
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	f, err := file.Open()
	if err != nil {
		dm.logger.Error("File open error:", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	defer f.Close()

	buf := make([]byte, file.Size)
	_, err = f.Read(buf)
	if err != nil {
		dm.logger.Error("File read error:", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	if _, err := dm.redisClient.HSet(c.Request.Context(), dm.resultHashKey, taskID, buf).Result(); err != nil {
		dm.logger.Error("Redis error:", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"success": true, "msg": "SUCCESS"})
}

func (dm *DistributedMaster) createTask(c *gin.Context) {
	file, err := c.FormFile("task_data")
	if err != nil {
		dm.logger.Error("File error:", err)
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	taskType := c.PostForm("task_type")
	if taskType != "cpu" && taskType != "gpu" {
		c.JSON(http.StatusInternalServerError, gin.H{
			"success": false,
			"msg":     "TASK Type ERROR",
		})
		return
	}

	taskID := c.PostForm("task_id")

	taskCost, err := strconv.ParseFloat(c.PostForm("task_cost"), 64)
	if err != nil {
		dm.logger.Error("Invalid task_cost:", err)
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid task_cost"})
		return
	}

	f, err := file.Open()
	if err != nil {
		dm.logger.Error("File open error:", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	defer f.Close()

	buf := make([]byte, file.Size)
	_, err = f.Read(buf)
	if err != nil {
		dm.logger.Error("File read error:", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	if _, err := dm.redisClient.HSet(c.Request.Context(), dm.taskDataHashKey, taskID, buf).Result(); err != nil {
		dm.logger.Error("Redis error:", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	if _, err := dm.redisClient.ZAdd(c.Request.Context(), taskType, &redis.Z{
		Score:  taskCost,
		Member: taskID,
	}).Result(); err != nil {
		dm.logger.Error("Redis error:", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"success": true, "msg": "SUCCESS"})
}

func (dm *DistributedMaster) showTaskRest() {
	defer dm.wg.Done()
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			ctx := context.Background()
			cpuCount, _ := dm.redisClient.ZCard(ctx, "cpu").Result()
			gpuCount, _ := dm.redisClient.ZCard(ctx, "gpu").Result()
			dm.logger.Infof("REST TASK NUM is %d", cpuCount+gpuCount)
		case <-dm.taskRestCtx.Done():
			return
		}
	}
}

func main() {
	port := 1088
	master := NewDistributedMaster(port)
	master.Run()
	for true {
		time.Sleep(1 * time.Second)
	}
}
