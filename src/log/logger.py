# -*- coding: utf-8 -*-
import sys
import logging
import threading
from pathlib import Path
from loguru import logger
from typing import Optional, Union
from src.log.generalConfig import GeneralConfig
from  datetime import datetime
import uuid

# --- 配置参数 ---
LOG_DIR = Path(GeneralConfig.logFilePath)
# 默认日志级别配置
DEFAULT_LOG_LEVEL = GeneralConfig.defaultLogLevel
LOG_FILE_PATH = ""
DEFAULT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<lvl>{level: <10}</lvl> | "
    "<cyan>{extra[module_name]:<15}</cyan>:<cyan>{function:<18}</cyan>:<yellow>{line:<5}</yellow> "
    "- <lvl>{message}</lvl>"
)

# Protect logger reconfiguration from concurrent access
_logLock = threading.Lock()


class LoggerManager:
    _initialized = False
    @classmethod
    def setup(
            cls,
            logFile: Optional[Union[str, Path]] = None,
            level: str = DEFAULT_LOG_LEVEL,
            rotation: str =GeneralConfig.rotation,
            retention: str = GeneralConfig.retention,
    ):
        if cls._initialized:
            return

        # 1. 移除默认配置
        logger.remove()

        # 2. 覆盖自定义 Level 的图标 (直接注入 loguru)
        # 这样在 format 中直接使用 {level.icon} 或 <lvl> 即可
        # 避免控制台编码问题，使用 ASCII 图标
        logger.level("TRACE", icon="🔍")
        logger.level("DEBUG", icon="🐛")
        logger.level("INFO", icon="ℹ️")
        logger.level("SUCCESS", icon="✅")
        logger.level("WARNING", icon="⚠️")
        logger.level("ERROR", icon="❌")
        logger.level("CRITICAL", icon="🔥")

        # 3. 添加控制台输出
        logger.add(
            sink=sys.stdout,
            format=cls._getFormat,
            colorize=True,
            level=level,
            backtrace=True,
            diagnose=True,
        )

        # 4. 添加文件输出
        if logFile:
            logPath = Path(logFile)
        else:
            # 使用默认的日志目录
            logPath = LOG_DIR /  f"{datetime.now().strftime('%Y-%m-%d-')}-{uuid.uuid4().hex}.log"
        
        logPath.parent.mkdir(parents=True, exist_ok=True)

        # 保存日志文件路径到全局变量
        global LOG_FILE_PATH
        LOG_FILE_PATH = logPath

        logger.add(
            sink=logPath,
            format=cls._getFileFormat,  # 文件输出使用更简洁的文本格式
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip",
            encoding="utf-8",
            enqueue=False,  # 避免多进程队列在受限环境下报错
        )

        # 5. 拦截标准 logging 库日志
        cls._interceptStandardLogging()

        cls._initialized = True
        logger.info(f"日志系统初始化完成，日志级别: {level}, 日志文件: {logPath}")

    @staticmethod
    def _getFormat(record):
        """
        动态构建控制台日志格式，利用已定义的 level icon
        """
        # 在 extra 中添加默认 module_name 防止报错
        if "module_name" not in record["extra"]:
            record["extra"]["module_name"] = record["name"]

        icon = record["level"].icon

        # 截断 module_name 和 function 字段，保持输出整齐
        maxModuleNameLength = 15
        maxFunctionLength = 18

        # 截断 module_name
        moduleName = record["extra"].get("module_name", "")
        if len(moduleName) > maxModuleNameLength:
            record["extra"]["module_name"] = moduleName[:maxModuleNameLength - 3] + "..."

        # 截断 function
        function = record.get("function", "")
        if len(function) > maxFunctionLength:
            record["function"] = function[:maxFunctionLength - 3] + "..."

        # 这里的 <lvl> 会自动根据级别应用 loguru 内置颜色
        return (
            "<bold><green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green></bold> "
            "<bold>│</bold> "
            f"<lvl>{icon} {{level: <8}}</lvl> "
            "<bold>│</bold> "
            "<cyan>{extra[module_name]:<15}</cyan>:<cyan>{function:<18}</cyan>:<yellow>{line:<10}</yellow> "
            "<bold>│</bold> "
            "<lvl>{message}</lvl>\n{exception}"
        )

    @staticmethod
    def _getFileFormat(record):
        """
        动态构建文件日志格式，不包含颜色代码
        """
        # 在 extra 中添加默认 module_name 防止报错
        if "module_name" not in record["extra"]:
            record["extra"]["module_name"] = record["name"]

        # 截断 module_name 和 function 字段，保持输出整齐
        maxModuleNameLength = 15
        maxFunctionLength = 18

        # 截断 module_name
        moduleName = record["extra"].get("module_name", "")
        if len(moduleName) > maxModuleNameLength:
            record["extra"]["module_name"] = moduleName[:maxModuleNameLength - 3] + "..."

        # 截断 function
        function = record.get("function", "")
        if len(function) > maxFunctionLength:
            record["function"] = function[:maxFunctionLength - 3] + "..."

        # 文件输出不使用颜色代码
        return (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <10} | "
            "{extra[module_name]:<15}:{function:<18}:{line:<10} | "
            "{message}\n{exception}"
        )

    @staticmethod
    def _interceptStandardLogging():
        """让标准库 logging 的日志也通过 loguru 输出，但过滤掉不需要的日志"""

        # 定义要过滤的日志来源（这些日志通常来自第三方库，比较嘈杂）
        FILTERED_LOGGERS = {
            'huggingface_hub',
            'urllib3',
            'requests',
            'httpx',
            'httpcore',
            'filelock',
            'PIL',
            'matplotlib',
            'tensorflow',
            'torch',
        }

        class InterceptHandler(logging.Handler):
            def emit(self, record):
                # 过滤特定来源的日志
                if record.name in FILTERED_LOGGERS or any(record.name.startswith(f"{prefix}.") for prefix in FILTERED_LOGGERS):
                    return
                
                # 过滤包含特定关键字的日志（如 HTTP 请求）
                message = record.getMessage()
                if 'HTTP Request:' in message or 'HEAD' in message or 'GET' in message:
                    return
                
                try:
                    level = logger.level(record.levelname).name
                except ValueError:
                    level = record.levelno
                frame, depth = logging.currentframe(), 2
                while frame.f_code.co_filename == logging.__file__:
                    frame = frame.f_back
                    depth += 1
                logger.opt(depth=depth, exception=record.exc_info).log(level, message)

        # 只拦截 WARNING 级别及以上的日志，减少噪音
        logging.basicConfig(handlers=[InterceptHandler()], level=logging.WARNING)


def getLogger(name: Optional[str] = None):
    """
    获取 logger。建议在每个文件开头使用：log = getLogger(__name__)
    """
    # 确保已经初始化（如果不手动调用 setup，这里可以给个默认值）
    if not LoggerManager._initialized:
        # 从配置中获取日志级别
        LoggerManager.setup(level=GeneralConfig.defaultLogLevel)

    # 使用 bind 将模块名绑定到 extra，方便 format 读取
    return logger.bind(module_name=name or "Main")


def setLogLevel(level: str):
    """
    动态设置日志级别（线程安全）

    Args:
        level: 日志级别，可选值：TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL
    """
    global LOG_FILE_PATH
    with _logLock:
        # 移除所有现有的处理器
        logger.remove()

        # 重新设置图标（保持与初始化一致的 ASCII 风格）
        logger.level("TRACE", icon="🔍")
        logger.level("DEBUG", icon="🐛")
        logger.level("INFO", icon="ℹ️")
        logger.level("SUCCESS", icon="✅")
        logger.level("WARNING", icon="⚠️")
        logger.level("ERROR", icon="❌")
        logger.level("CRITICAL", icon="🔥")

        # 重新添加控制台输出
        logger.add(
            sink=sys.stdout,
            format=LoggerManager._getFormat,
            colorize=True,
            level=level,
            backtrace=True,
            diagnose=True,
        )

        # 重新添加文件输出（如果之前有）
        if LOG_FILE_PATH:
            logger.add(
                sink=LOG_FILE_PATH,
                format=LoggerManager._getFileFormat,
                level=level,
                rotation="10 MB",
                retention="7 days",
                compression="zip",
                encoding="utf-8",
                enqueue=True,
            )

    logger.info(f"日志级别已设置为: {level}")

# 初始化日志系统
LoggerManager.setup()
