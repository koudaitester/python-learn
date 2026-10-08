from urllib.parse import unquote, quote

def url_decode(raw_str: str) -> str:
    """URL解码：%xx 转原始字符"""
    return unquote(raw_str)

def url_encode(raw_str: str) -> str:
    """URL编码：特殊字符转%xx"""
    return quote(raw_str, safe='')
