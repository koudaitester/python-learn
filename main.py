from fastapi import FastAPI
import psycopg2
from pydantic import BaseModel

app = FastAPI()


# 数据库连接信息，填你Dify容器PG的地址账号
DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5432,
    "user": "dify",
    "password": "Difyai123456",
    "database": "dify"
}

# 入库请求体模型
class InboundItem(BaseModel):
    商品名: str
    规格: str
    分类: str
    货架号: str
    进价: float
    售价: float
    库存: int

# 入库接口
@app.post("/inbound")
def create_inbound(item: InboundItem):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    sql = """
    INSERT INTO goods_inbound_record ("商品名", "规格", "分类", "货架号", "进价", "售价", "库存")
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    RETURNING id;
    """
    cur.execute(sql, (item.商品名, item.规格, item.分类, item.货架号, item.进价, item.售价, item.库存))
    new_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return {"code":0, "msg":"入库成功", "id": new_id}

# 查询接口
@app.get("/query")
def query_goods(货架号:str=None, 商品名:str=None):
    print(f"test query 1---------")
    conn = psycopg2.connect(**DB_CONFIG)
    print(f"test query 2---------")
    
    cur = conn.cursor()
    print(f"test connection------------{conn}, cur==========={cur}")
    sql = 'SELECT * FROM goods_inbound_record WHERE 1=1'
    params = []
    if 货架号:
        sql += ' AND "货架号" = %s'
        params.append(货架号)
    if 商品名:
        sql += ' AND "商品名" LIKE %s'
        params.append(f'%{商品名}%')
    cur.execute(sql, params)
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    result = [dict(zip(columns, r)) for r in rows]
    cur.close()
    conn.close()
    return {"code":0, "data":result}

if __name__ == "__main__":
    import uvicorn
    # host改成0.0.0.0
    uvicorn.run("main:app", host="0.0.0.0", port=6100, reload=True)

