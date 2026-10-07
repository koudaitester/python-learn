from services.logger_service import logger

logger.info("测试INFO日志")
logger.warning("测试WARN日志")
logger.error("测试ERROR日志")
print("日志写入完成，查看控制台和 logs/app.log")
