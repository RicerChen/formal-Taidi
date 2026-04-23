import os
import time
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from typing import List, Dict, Any, Literal, Annotated
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from sqlalchemy import create_engine
import re

# 配置API密钥
os.environ["OPENAI_API_KEY"] = "sk-19fe6b9376f8473bab1defd0bde82559"

# 初始化LLM
llm = ChatOpenAI(
    model="kimi-k2.5",
    temperature=0,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 数据处理函数
def clean_text(text):
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def normalize_text(s):
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    s = s.replace("（", "(").replace("）", ")")
    s = s.replace("：", ":")
    s = s.replace("，", ",")
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("‘", "'").replace("’", "'")
    s = re.sub(r"\s+", " ", s).strip()
    return s

# 数据库类型映射
TABLE_DTYPE_MAP = {
    "core": {
        "stock_code": "String(6)",
    },
    "balance": {
        "stock_code": "String(6)",
    },
    "cash": {
        "stock_code": "String(6)",
    },
    "income": {
        "stock_code": "String(6)",
    },
}

# 初始化数据库
def init_database():
    engine = create_engine("sqlite:///example.db")
    
    def normalize_stock_code(x):
        if pd.isna(x):
            return None
        s = str(x).strip()
        if not s:
            return None
        s = re.sub(r"\.0+$", "", s)
        if s.isdigit():
            s = s.zfill(6)
        return s
    
    def cast_df_by_table(df, table_name):
        df = df.copy()
        dtype_map = TABLE_DTYPE_MAP.get(table_name, {})
        
        for col, sql_type in dtype_map.items():
            if col not in df.columns:
                continue
            if col == "stock_code":
                df[col] = df[col].apply(normalize_stock_code).astype("string")
            else:
                df[col] = df[col].astype("string")
        return df
    
    # 读取Excel文件
    sheets = pd.read_excel(
        r"final_result_final.xlsx",
        sheet_name=None,
        engine="openpyxl"
    )
    
    for sheet_name, df in sheets.items():
        table_name = sheet_name.strip().replace(" ", "_").replace("-", "_")
        df = cast_df_by_table(df, table_name)
        df.to_sql(
            table_name,
            con=engine,
            if_exists="replace",
            index=False
        )
    
    return SQLDatabase.from_uri("sqlite:///example.db")

# 状态定义
class ParentState:
    def __init__(self):
        self.messages = []
        self.query = ""
        self.real_query = ""
        self.route = ""
        self.reason = ""
        self.missing_slots = []
        self.need_sql = False
        self.need_clarification = False
        self.sql_result = {}
        self.final_answer = ""
        self.summary = ""
        self.sql_query = ""
        self.sql_answer = ""
        self.sql_messages = []
        self.question_count = 0
        self.skip_summary = False

# 工具节点
def normalize_tool_call(tool_call):
    tc = dict(tool_call or {})
    raw_name = str(tc.get("name") or "").strip()
    args = tc.get("args", {}) or {}
    
    if isinstance(args, str):
        args = {"query": args}
    
    parts = [p for p in raw_name.split("\x00") if p != ""]
    base_name = parts[0] if parts else raw_name
    
    if base_name.startswith("sql_db_query"):
        if len(parts) >= 3 and not args:
            key = parts[1].strip()
            value = "\x00".join(parts[2:]).strip()
            if key:
                args = {key: value}
        base_name = "sql_db_query"
    elif base_name.startswith("sql_db_schema"):
        base_name = "sql_db_schema"
    elif base_name.startswith("sql_db_list_tables"):
        base_name = "sql_db_list_tables"
    
    tc["name"] = base_name
    tc["args"] = args
    return tc

# 路由决策
class RouteDecision(BaseModel):
    route: Literal["sql", "clarify"] = Field(
        description="The routing decision for the user query."
    )
    reason: str = Field(
        description="One-sentence reason for the routing decision."
    )
    missing_slots: List[str] = Field(
        default_factory=list,
        description="Missing required slots such as company, period, metric. Empty list if none."
    )

def route_node(state):
    question = state.real_query
    
    route_prompt = f"""
    你是一个上市公司财报智能问数助手的“路由器（Router）”。
    
    你的任务不是回答用户问题，而是判断这个问题应该走哪条处理链路。
    
    你只能输出以下两种 route：
    1. sql
       - 适用于可以主要通过结构化财务数据库回答的问题
       - 典型包括：数值查询、指标查询、同比/环比、趋势、排名、topN、筛选、排序、聚合统计、跨公司比较
    2. clarify
       - 适用于用户问题缺少关键条件，当前无法可靠执行
       - 例如缺少公司名、报告期、指标名、比较对象等
    
    判定原则：
    - 如果问题主要在问“多少、是否、排名、变化趋势、同比环比、topN、哪个最大/最小”，优先判为 sql
    - 如果问题缺少关键查询条件，判为 clarify
    
    请特别注意：
    - “趋势分析”“同比分析”“排名分析”通常仍然属于 sql
    
    示例1
    用户问题：比亚迪2025年三季度营业收入是多少？
    输出：
    {{
      "route": "sql",
      "reason": "这是明确的单指标单时期数值查询，可直接通过结构化财务数据库回答",
      "missing_slots": [],
    }}
    
    示例2
    用户问题：金花股份近几年的利润总额变化趋势是什么样的？
    输出：
    {{
      "route": "sql",
      "reason": "这是时间序列趋势分析问题，主要依赖结构化财务数据",
      "missing_slots": [],
    }}
    
    示例3
    用户问题：利润总额是多少？
    输出：
    {{
      "route": "clarify",
      "reason": "缺少公司和报告期，无法直接执行查询",
      "missing_slots": ["company", "period"],
    }}
    
    示例4
    用户问题：华为每股收益是多少？
    输出：
    {{
      "route": "clarify",
      "reason": "缺少报告期，无法直接执行查询",
      "missing_slots": ["period"],
    }}
    
    现在请判断下面这个用户问题：
    用户问题：{question}
    
    请严格输出 JSON，不要输出任何额外解释：
    
    {{
      "route": "sql | clarify",
      "reason": "一句话说明分类原因",
      "missing_slots": [],
    }}
    """.format(question=question)
    
    structured_llm = llm.with_structured_output(RouteDecision)
    try:
        decision = structured_llm.invoke(route_prompt)
        state.route = decision.route
        state.reason = decision.reason
        state.missing_slots = decision.missing_slots
        state.need_sql = decision.route == "sql"
        state.need_clarification = decision.route == "clarify"
    except Exception as e:
        print(f"路由决策错误: {e}")
        state.route = "clarify"
        state.reason = "路由决策失败"
        state.missing_slots = ["unknown"]
        state.need_sql = False
        state.need_clarification = True
    
    return state

# 澄清函数
def clarify(state):
    question = state.real_query
    missing_slots = state.missing_slots
    
    clarify_prompt = f"""
        你是一个上市公司财报智能问数助手。
        
        用户原始问题：
        {question}
        
        当前还缺少这些关键信息：
        {missing_slots}
        
        你的任务是：
        基于用户原始问题和缺失槽位，生成一句自然、简洁、礼貌的澄清问题，引导用户一次性补全所缺信息，以便继续查询。
        
        要求：
        1. 只输出一句面向用户的澄清问题，不要输出解释，不要输出 JSON，不要输出多余内容。
        2. 如果缺少多个槽位，尽量合并成一句话一起问，不要拆成多句。
        3. 语气自然，适合中文对话场景。
        4. 问法要贴合财报问数场景。
        
        示例1
        原始问题：利润总额是多少？
        缺失槽位：["company", "period"]
        输出：请问你想查询哪家公司，以及哪个报告期的利润总额？
        
        示例2
        原始问题：同比最高的是哪家？
        缺失槽位：["metric", "period", "ranking_scope"]
        输出：请问你想查询哪个指标、哪个报告期，以及希望在哪个范围内比较同比？
        
        示例3
        原始问题：原因是什么？
        缺失槽位：["explanation_target"]
        输出：请问你具体想了解哪个指标或现象变化的原因？
        """.strip()
    
    try:
        response = llm.invoke([{"role": "user", "content": clarify_prompt}])
        state.final_answer = response.content
    except Exception as e:
        print(f"澄清错误: {e}")
        state.final_answer = "抱歉，我无法理解您的问题，请提供更多信息。"
    
    return state

# SQL查询函数
def run_sql_query(state, db):
    query = state.real_query
    
    # 初始化工具
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
    run_query_tool = next(tool for tool in tools if tool.name == "sql_db_query")
    
    # 生成SQL查询
    generate_query_system_prompt = """
    你是一个专为与 SQL 数据库交互而设计的智能体。
    给定一个用户输入问题，请生成一条语法正确的 sqlite 查询语句并执行，
    然后查看查询结果并返回答案。除非用户明确指定了希望获取的示例数量，否则始终将你的查询结果限制在最多 5 条。
    
    你可以按相关列对结果进行排序，以返回数据库中最有趣的示例。切勿查询特定表的所有列，仅根据问题请求相关的列。
    
    对于趋势/可视化类问题（例如：趋势、变化、走势、近几年、近年、历年、近年来、多年、可视化、绘图、折线图、柱状图）：
    1. 返回时间序列查询，而非单个标量结果。
    2. 选择一个时间列和一个数值指标列。
    3. 优先保留原始字段名，除非确有必要，否则不要统一别名为 x_axis 和 y_axis。
    4. 按时间顺序升序排列结果。
    5. 不要机械地应用默认的 top_k 限制；应保留绘图所需的所有时间段。
    
    【时间粒度选择规则】
    A. 如果问题是“跨年度趋势”，例如包含：
    “近几年、近年、历年、近年来、多年”，并且语义是在问整体趋势/变化/走势，
    那么：
    - 优先使用 report_year 作为时间轴；
    - 如果表中存在 report_period 字段，则只保留年度口径数据；
    - 年度口径优先匹配 FY、年度、年报、12-31、1231、年末 这类含义；
    - 不要同时返回 Q1、HY、Q3、FY 等多种期间；
    - 最终结果必须保证每个 report_year 最多一条记录。
    
    B. 如果问题是“某一年内的季度/期间变化”，例如包含：
    “各季度、分季度、季度变化、Q1、Q3、半年报、三季报、某年内变化”，
    那么：
    - 不要只用 report_year 作为时间轴；
    - 应优先使用 report_period，必要时将 report_year 和 report_period 拼接成一个唯一时间标签；
    - 结果可按 Q1 < H1 < Q3 < FY 的顺序排序；
    - 同一时间标签只能保留一条记录。
    
    C. 如果用户明确说“按年度看”“年度趋势”，按规则 A 处理。
    D. 如果用户明确说“按季度看”“分季度看”，按规则 B 处理。
    
    【指标字段优先级规则】
    1. 当用户明确问“同比增长率”“环比增长率”“增长率”时：
       - 如果 schema 中存在对应的现成增长率字段，必须优先直接查询该字段；
       - 不要改查原始金额字段后再自行计算。
    2. 典型映射如下：
       - 营业总收入环比增长率 / 营业收入环比增长率 / 营收环比增长率 -> operating_revenue_qoq_growth
       - 营业总收入同比增长率 / 营业收入同比增长率 / 营收同比增长率 -> operating_revenue_yoy_growth
       - 净利润环比增长率 / 归母净利润环比增长率 -> net_profit_qoq_growth
       - 净利润同比增长率 / 归母净利润同比增长率 -> net_profit_yoy_growth
       - 总资产同比增长率 -> asset_total_assets_yoy_growth
       - 总负债同比增长率 -> liability_total_liabilities_yoy_growth
    3. 只有当对应增长率字段不存在时，才允许改查原始金额字段并自行计算。
    
    【累计口径转单季规则】
    1. 财报中的 HY、FY 往往是累计值，不一定是单季值。
    2. 若必须根据原始金额自行计算单季值，必须使用以下规则：
       - Q1单季 = Q1累计
       - Q2单季 = H1累计 - Q1累计
       - Q3单季 = Q3累计
       - Q4单季 = FY累计 - H1累计 - Q3累计
    3. 若问题要求“环比增长率”，必须先转成单季值，再计算环比：
       - Q2环比 = (Q2单季 - Q1单季) / Q1单季
       - Q3环比 = (Q3单季 - Q2单季) / Q2单季
       - Q4环比 = (Q4单季 - Q3单季) / Q3单季
    4. 禁止把累计值直接拿去和上一季度累计值做环比。
    
    【显式期间过滤规则】
    1. 如果问题明确指定了目标期间，例如“2025年第三季度与2024年第三季度相比”，最终返回结果必须只保留这两个目标期间对应的结果。
    2. 如果为了中间计算需要额外用到 Q1、H1、Q3、FY 等期间，可以在 CTE 或子查询中使用；
       但最终 SELECT 不要把无关期间（如 FY、Q1）返回给用户。
    3. 对于“请展示两年同期的环比数据”这类问题，最终结果应类似：
       report_year, report_period, operating_revenue_qoq_growth
       2024, Q3, ...
       2025, Q3, ...
    
    【结果唯一性要求】
    - 若时间轴是 report_year，则每个 report_year 只能出现一行。
    - 若时间轴是 report_period 或拼接时间标签，则每个时间标签只能出现一行。
    - 如底层原始表中同一年存在多条候选记录，优先保留年度口径；若无年度口径，再保留该年最后一个可用期间。
    
    切勿对数据库执行任何 DML 语句（如 INSERT、UPDATE、DELETE、DROP 等）。
    
    【股票代码字段规则】
    - stock_code 一律按字符串处理，不要按数字处理。
    - 查询股票代码时，必须使用带引号的 6 位代码，例如 '000538'、'600436'。
    - 不要写成 stock_code = 538 或 stock_code = 600436。
    - 如果结果中需要返回股票代码，优先直接返回原始 stock_code 字段，不要做数值运算。
    
    如果上下文中已经给出了所需表的 schema，请不要再次调用 sql_db_schema 或 sql_db_list_tables。
    此阶段只允许两种行为：
    1. 生成并调用 sql_db_query；
    2. 在已有查询结果时直接输出最终答案。
    
    【最终回答格式要求】
    1. 只输出纯文本，不要使用 Markdown。
    2. 不要使用表格语法，如 |---|、| 列 |。
    3. 不要使用 **粗体**、# 标题、- 列表。
    4. 严禁输出字面量字符“\\n”、“\\t”以及任何其他转义字符。
    5. 先给结论，再给必要说明。
    6. 输出内容要适合直接写入 Excel 单元格。
    7. 如果问题属于趋势/变化/走势/历年分析类，不要只给一句笼统结论，必须结合查询结果展开说明。
    8. 对于趋势类问题，回答中尽量包含以下信息：
       总体趋势判断；各时间段关键数值变化；相邻时期增减变化；全周期累计变化幅度；如连续增长或连续下降要明确指出。
    9. 如果查询结果是多期时间序列，回答不少于3句，避免只输出一句压缩总结。
    10. 回答仍然使用纯文本，但可以分句清晰表达，不要过度简写。
    11. 严禁输出任何思考过程、推理过程、自言自语、计划性语句。
    12. 严禁出现类似：
        Now I have..., Let me..., I will..., First, ...
        我先..., 下面我来..., 让我先..., 接下来我...
    13. 不要描述你正在查看数据或接下来要做什么，只直接输出面向用户的最终答案。
    """
    
    # 获取数据库schema
    schema_prompt = "请提供所有表的schema信息"
    llm_with_schema = llm.bind_tools([get_schema_tool], tool_choice="any")
    schema_response = llm_with_schema.invoke([HumanMessage(content=schema_prompt)])
    
    # 生成SQL查询
    sql_prompt = f"""
    数据库schema：
    {schema_response.content}
    
    用户问题：{query}
    
    请生成SQL查询语句并执行，然后返回答案。
    """
    
    llm_with_query = llm.bind_tools([run_query_tool], tool_choice="any")
    try:
        sql_response = llm_with_query.invoke([HumanMessage(content=sql_prompt)])
        # 提取SQL查询结果
        if hasattr(sql_response, "tool_calls") and sql_response.tool_calls:
            tool_call = sql_response.tool_calls[0]
            normalized_call = normalize_tool_call(tool_call)
            if normalized_call["name"] == "sql_db_query":
                state.sql_query = normalized_call["args"].get("query", "")
                # 执行查询
                from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
                query_tool = QuerySQLDataBaseTool(db=db)
                sql_result = query_tool.run(state.sql_query)
                state.sql_answer = sql_result
                # 生成自然语言回答
                answer_prompt = f"""
                SQL查询结果：
                {sql_result}
                
                用户问题：{query}
                
                请将查询结果转换为自然、友好的中文回答。
                """
                answer_response = llm.invoke([HumanMessage(content=answer_prompt)])
                state.final_answer = answer_response.content
            else:
                state.final_answer = "无法生成SQL查询"
        else:
            state.final_answer = sql_response.content
    except Exception as e:
        print(f"SQL查询错误: {e}")
        state.final_answer = f"查询出错: {str(e)}"
    
    return state

# 主处理函数
def process_query(query, db):
    start_time = time.time()
    
    state = ParentState()
    state.query = query
    state.real_query = query
    
    # 路由决策
    state = route_node(state)
    
    if state.need_sql:
        state = run_sql_query(state, db)
    else:
        state = clarify(state)
    
    end_time = time.time()
    response_time = end_time - start_time
    
    return {
        "answer": state.final_answer,
        "response_time": response_time,
        "sql_query": state.sql_query,
        "route": state.route
    }

# 测试函数
def run_tests():
    # 初始化数据库
    print("初始化数据库...")
    db = init_database()
    print("数据库初始化完成")
    
    # 测试用例
    test_cases = [
        "华润三九2025年第三季度的利润总额是多少？",
        "金花股份近几年的利润总额变化趋势是什么样的？",
        "2024年利润最高的top10企业是哪些？",
        "这些企业的利润、销售额年同比是多少？",
        "利润最高的企业2024年的毛利率是多少？"
    ]
    
    # 存储结果
    results = []
    
    # 运行测试
    print("\n开始测试...")
    for i, test_case in enumerate(test_cases):
        print(f"\n测试用例 {i+1}: {test_case}")
        result = process_query(test_case, db)
        results.append({
            "test_case": test_case,
            "response_time": result["response_time"],
            "answer": result["answer"],
            "sql_query": result.get("sql_query", ""),
            "route": result.get("route", "")
        })
        print(f"响应时间: {result['response_time']:.2f}秒")
        print(f"路由: {result.get('route', 'N/A')}")
        print(f"结果: {result['answer'][:100]}...")  # 只显示前100个字符
        print("-" * 80)
    
    # 计算平均响应时间
    avg_time = sum(r["response_time"] for r in results) / len(results)
    print(f"\n平均响应时间: {avg_time:.2f}秒")
    
    # 保存结果到Excel
    df = pd.DataFrame(results)
    df.to_excel("实验结果.xlsx", index=False)
    print("实验结果已保存到 实验结果.xlsx")
    
    return results

if __name__ == "__main__":
    run_tests()
