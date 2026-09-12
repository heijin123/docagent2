"""扩充 M6 评估语料到 data/samples/（与 make_samples.py 分工：本脚本只管"加量"）。

为什么需要它
------------
评估指标 recall@5 的分母是**整个 chunk 语料**。原有语料仅 5 个文档 / 约 11 个
chunk，top-5 一把就能覆盖近半，任何检索器都接近满分——指标失去区分度。
本脚本追加 13 个文档，把语料撑到"top-5 只覆盖一小部分"的量级，让 recall@5 / MRR
真正能反映检索质量。

干扰项设计（关键，不是随便塞文件）
--------------------------------
- **强干扰（hard negative）**：与已有文档"语义高度相似但结论冲突"，检索器必须
  分辨版本/型号才能选对。例如：
    - 差旅报销 v1（2025）住宿 500/天、交通 80/天、审批线 5000
      vs v2（2026）住宿 800/天、交通 120/天、审批线 8000；
    - Guardian X1 与 Sentinel S2 两份产品手册章节结构几乎一致、参数不同；
    - 年假规定（跨年累计 10 天）vs 已有《员工休假制度》（5 天）。
- **弱干扰（easy negative）**：食堂菜单、团建通知、停车指引、入职指引，模拟真实
  知识库的噪声，只负责把分母做大。

格式覆盖：md / txt / docx / pdf 各有新样本；PDF 含**恒定页眉页脚**（跨页重复），
用于验证 parsers.py 的页眉页脚剔除逻辑；文件名含日期（如 2025-03-01）用于验证
F2.9 文件名日期抽取。

幂等：重复运行只覆盖同名文件，不会删除既有语料（含 make_samples.py 产出的 5 个）。

用法::

    python scripts/make_corpus.py
    python scripts/make_corpus.py --out data/samples
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DEFAULT_OUT = BASE / "data" / "samples"

# ══════════════════════════════════════════════════════════════════════
# Markdown 语料
# ══════════════════════════════════════════════════════════════════════

TRAVEL_V1_MD = """# 差旅费用报销管理制度（2025 版）

## 一、适用范围

本制度适用于全体正式员工因公出差产生的交通、住宿与餐饮费用报销。

## 二、住宿标准

| 职级 | 城市类别 | 住宿上限 | 凭证要求 |
|---|---|---|---|
| 普通员工 | 一线城市 | 500/天 | 需纸质发票 |
| 普通员工 | 二线城市 | 350/天 | 需纸质发票 |
| 部门经理 | 一线城市 | 700/天 | 需纸质发票 |

## 三、市内交通

市内交通费实行定额补贴，每人每天 80 元，无需提供票据。超出部分由个人承担。

## 四、审批权限

单次差旅报销总额在 5000 元以内的，由部门经理审批；超过 5000 元的，须报总监审批。

## 五、报销时限

出差结束后 30 个自然日内提交报销单，逾期不予受理。

## 六、附则

本制度自 2025 年 3 月 1 日起执行。
"""

TRAVEL_V2_MD = """# 差旅费用报销管理制度（2026 修订版）

## 一、适用范围

本制度适用于全体正式员工及签订劳务合同的外部顾问因公出差产生的费用报销。

## 二、住宿标准

| 职级 | 城市类别 | 住宿上限 | 凭证要求 |
|---|---|---|---|
| 普通员工 | 一线城市 | 800/天 | 电子发票即可 |
| 普通员工 | 二线城市 | 550/天 | 电子发票即可 |
| 部门经理 | 一线城市 | 1100/天 | 电子发票即可 |

## 三、市内交通

市内交通费实行定额补贴，每人每天 120 元，无需提供票据。

## 四、审批权限

单次差旅报销总额在 8000 元以内的，由部门经理审批；超过 8000 元的，须报总监审批。

## 五、报销时限

出差结束后 15 个自然日内提交报销单，逾期需附情况说明并由总监签字。

## 六、新旧制度衔接

本制度自 2026 年 1 月 1 日起执行，2025 版《差旅费用报销管理制度》同时废止。
"""

ANNUAL_LEAVE_MD = """# 年假管理规定（2026 修订）

## 一、适用范围

本规定适用于全体正式员工，试用期员工不适用。

## 二、年假天数

入职满 1 年不满 10 年的员工，每年享有 5 天年假。
入职满 10 年不满 20 年的员工，每年享有 10 天年假。
入职满 20 年以上的员工，每年享有 15 天年假。

## 三、跨年结转

年假按自然年计算，可跨年累计不超过 10 天，超出部分自动作废。

## 四、申请流程

年假需提前 3 个工作日在系统提交申请，由直属主管与人力资源部双重审批。

## 五、未休年假折算

因工作原因未休完的年假，可申请折算为工资，折算标准为日工资的 200%。

## 六、附则

本规定自 2026 年 1 月 1 日起执行，最终解释权由人力资源部行使。
"""

PARKING_MD = """# 园区停车管理规定

## 一、车位分配

园区共有地面车位 120 个、地下车位 260 个。
员工车位按部门统一分配，访客车位设在地面 A 区，共 20 个。

## 二、停车费用

员工月卡 150 元，季度卡 400 元，年卡 1500 元。
访客车辆前 2 小时免费，超出部分每小时 5 元。

## 三、管理规定

车辆须按标线停放，禁止占用消防通道与无障碍车位。
园区内限速 10 公里每小时，禁止鸣笛。

## 四、违规处理

违停一次予以警告，累计三次取消当月月卡资格。
"""

ONBOARDING_MD = """# 新员工入职指引

## 一、入职当天需携带材料

身份证原件及复印件、学历证书、离职证明、一寸照片两张。

## 二、入职办理流程

第一步：前往 B 座三层人力资源部报到并签署劳动合同。
第二步：领取工牌、办公用品与门禁卡。
第三步：由 IT 部门开通邮箱与内部系统账号。
第四步：参加为期半天的入职培训。

## 三、试用期安排

试用期为 3 个月，试用期考核合格后转正。

## 四、常用联系人

人力资源部对接人分机 8012，IT 服务台分机 8000。
"""

LONG_HANDBOOK_MD = """# 客户服务手册（2026 版）

## 第一章 服务范围

本手册适用于公司销售的全部硬件产品与配套软件服务的售后支持工作。
服务内容包括故障报修、远程协助、上门维修、备件更换与定期巡检五类。
不包含因人为损坏、私自拆机、非授权改装导致的故障，此类情形按有偿维修处理。

## 第二章 服务等级与响应时效

公司按客户签约等级提供差异化服务，等级由年度采购额自动核定。
钻石级客户享受 7×24 小时热线支持，报修后 2 小时内响应，4 小时内到场。
黄金级客户享受 5×8 小时热线支持，报修后 4 小时内响应，次日到场。
白银级客户享受 5×8 小时热线支持，报修后 8 小时内响应，两个工作日内到场。
未签约的散客按次提供服务，不承诺响应时效。

## 第三章 报修流程

第一步，客户通过服务热线提交报修，或经企业微信服务号自助下单。
第二步，客服在 30 分钟内完成工单登记并生成工单编号，编号规则为 SH 加 8 位数字。
第三步，工程师联系客户确认故障现象，必要时先进行远程诊断以缩短停机时间。
第四步，现场处理完成后由客户在工单上签字确认，未签字工单不得闭环。
第五步，客服在 24 小时内进行满意度回访，回访结果计入工程师绩效考核。

## 第四章 备件政策

保修期内的非人为故障，备件由公司免费提供，客户无需承担物流费用。
保修期外的维修，备件按官方价目表收费，人工费按工时另计。
更换下的旧件归公司所有，客户如需保留须在维修前提交书面申请。
关键备件的区域间调拨时限不超过 48 小时，超时须向客户书面说明原因。

## 第五章 保修期限

整机保修期为自交付之日起 24 个月，以交付签收单日期为准。
电池、风扇、密封圈等易损耗部件保修期为 12 个月，不在整机保修范围内。
软件服务包含 12 个月免费升级，到期后可续订年度服务包。
因自然灾害、供电异常导致的产品损坏不在保修范围内。

## 第六章 收费标准

上门服务费按城区 200 元每次、郊区 400 元每次收取，同址多台设备只计一次。
远程协助服务费按 100 元每小时收取，不足一小时按一小时计。
夜间与法定节假日服务加收 50% 的附加费，附加费在工单中单独列示。
所有收费项目须在服务前向客户书面报价并取得确认。

## 第七章 定期巡检与培训

钻石级客户每季度提供一次免费巡检，黄金级客户每半年一次。
巡检内容包括除尘、紧固、固件版本核查与性能测试，出具巡检报告。
公司每年为签约客户提供两次免费操作培训，培训地点与形式由双方协商确定。

## 第八章 投诉与升级

客户对服务结果不满意，可向服务经理投诉，服务经理应在 1 个工作日内给出处理意见。
若客户仍不满意，可升级至客户服务中心总监，总监在 3 个工作日内答复。
对超时未响应的工单，客户有权要求减免本次服务费用。

## 第九章 数据与保密

工程师在服务过程中接触到的客户数据仅限本次维修使用，不得复制或外传。
涉及数据销毁的服务须由客户现场监督并签署确认单。
服务工单相关的记录保存期为 3 年。

## 第十章 附则

本手册自 2026 年 4 月 1 日起执行。
手册内容如有调整，以公司官网公布的最新版本为准，恕不另行通知。
"""

# ══════════════════════════════════════════════════════════════════════
# 纯文本语料（parser 按空行切段，故段落间必须留空行）
# ══════════════════════════════════════════════════════════════════════

CANTEEN_TXT = """员工食堂一周菜单

周一：红烧排骨、清炒时蔬、紫菜蛋花汤。
周二：番茄牛腩、蒜蓉西兰花、冬瓜排骨汤。
周三：宫保鸡丁、手撕包菜、玉米浓汤。
周四：清蒸鲈鱼、干煸四季豆、银耳莲子羹。
周五：黑椒牛柳、上汤娃娃菜、酸辣汤。

供餐时间为早餐 07:30 至 09:00，午餐 11:30 至 13:30，晚餐 17:30 至 19:00。
夜班加餐仅在旺季开放，需提前一天在前台登记。

食堂位于 B 座一层，可容纳 300 人同时用餐。
饭卡充值请前往行政前台办理，支持现金与扫码两种方式。
"""

TEAM_BUILDING_TXT = """关于开展 2026 年秋季团建活动的通知

各部门：

为增强团队凝聚力，公司定于 10 月 24 日至 10 月 25 日组织秋季团建活动。

活动地点为千岛湖拓展基地，统一乘坐大巴前往，集合时间为 10 月 24 日 07:00。
集合地点为园区北门停车场，请提前十分钟到场签到。

请各部门于 10 月 15 日前将参加人员名单提交至行政部邮箱。
名单提交后如需变更，请说明原因并经部门负责人确认。

活动期间请穿着运动服装与防滑鞋，患有高血压、心脏病的同事请提前报备。
本次活动费用由公司承担，个人消费部分自理。
"""

# ══════════════════════════════════════════════════════════════════════
# DOCX 语料
# ══════════════════════════════════════════════════════════════════════

def _cjk_font() -> str | None:
    """PyMuPDF 默认字体不支持中文：优先用系统中文字体，否则 None（仅 ASCII）。"""
    import os

    for candidate in (
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def gen_travel_v1(path: Path) -> None:
    path.write_text(TRAVEL_V1_MD, encoding="utf-8")


def gen_travel_v2(path: Path) -> None:
    path.write_text(TRAVEL_V2_MD, encoding="utf-8")


def gen_annual_leave(path: Path) -> None:
    path.write_text(ANNUAL_LEAVE_MD, encoding="utf-8")


def gen_parking(path: Path) -> None:
    path.write_text(PARKING_MD, encoding="utf-8")


def gen_onboarding(path: Path) -> None:
    path.write_text(ONBOARDING_MD, encoding="utf-8")


def gen_long_handbook(path: Path) -> None:
    path.write_text(LONG_HANDBOOK_MD, encoding="utf-8")


def gen_canteen(path: Path) -> None:
    path.write_text(CANTEEN_TXT, encoding="utf-8")


def gen_team_building(path: Path) -> None:
    path.write_text(TEAM_BUILDING_TXT, encoding="utf-8")


def gen_sentinel_manual(path: Path) -> None:
    """DOCX：与 Guardian X1 章节结构一致、参数不同 → strong negative。"""
    import docx

    doc = docx.Document()
    doc.add_heading("Sentinel S2 手持终端 用户手册", level=1)

    doc.add_heading("一、产品概述", level=2)
    doc.add_paragraph(
        "Sentinel S2 是一款面向仓储场景的手持数据采集终端，支持一维码与二维码扫描。"
        "设备防护等级为 IP65，工作温度范围为零下 10 摄氏度至 50 摄氏度。"
    )

    doc.add_heading("二、技术参数", level=2)
    table = doc.add_table(rows=5, cols=2)
    table.style = "Table Grid"
    for r, (k, v) in enumerate([
        ("处理器", "八核 2.0GHz"),
        ("内存 / 存储", "6GB / 128GB"),
        ("电池容量", "4500mAh"),
        ("连续作业续航", "约 8 小时"),
    ]):
        table.cell(r, 0).text = k
        table.cell(r, 1).text = v
    doc.add_paragraph("无线连接支持 Wi-Fi 5 与蓝牙 5.0。")

    doc.add_heading("三、使用说明", level=2)
    doc.add_paragraph("开机后长按电源键 5 秒进入主界面。")
    doc.add_paragraph("首次使用需完成设备绑定，绑定码为 6 位数字。")
    doc.add_paragraph("扫描数据默认每 30 分钟自动同步一次。")

    doc.add_heading("四、维护保养", level=2)
    doc.add_paragraph("每 12 个月更换一次防尘滤网。")
    doc.add_paragraph("设备跌落或进水后应停止使用并送修，禁止自行拆解。")

    doc.save(str(path))


def gen_procurement_flow(path: Path) -> None:
    """DOCX：与 sample_guide.md 的《报销规范》抢"审批/金额"语义 → strong negative。"""
    import docx

    doc = docx.Document()
    doc.add_heading("采购付款管理流程", level=1)

    doc.add_heading("一、目的", level=2)
    doc.add_paragraph("规范公司采购活动中的申请、审批与付款环节，降低资金占用与合规风险。")

    doc.add_heading("二、适用范围", level=2)
    doc.add_paragraph("本流程适用于生产物料、办公用品与服务类采购。固定资产采购另按资产管理办法执行。")

    doc.add_heading("三、金额分级审批", level=2)
    table = doc.add_table(rows=4, cols=2)
    table.style = "Table Grid"
    for r, (k, v) in enumerate([
        ("采购金额区间", "审批人"),
        ("1 万元以内", "部门经理"),
        ("1 万元至 10 万元", "分管副总"),
        ("10 万元以上", "总经理办公会决议"),
    ]):
        table.cell(r, 0).text = k
        table.cell(r, 1).text = v

    doc.add_heading("四、付款方式", level=2)
    doc.add_paragraph("原则上采用对公转账，付款周期为验收合格后 45 个自然日。")
    doc.add_paragraph("单笔超过 5 万元的付款需附供应商银行账户信息复核单。")

    doc.add_heading("五、供应商管理", level=2)
    doc.add_paragraph("新供应商需完成资质审查与现场考察，纳入合格供应商名录后方可下单。")

    doc.add_heading("六、附则", level=2)
    doc.add_paragraph("本流程自 2026 年 2 月 1 日起执行。")

    doc.save(str(path))


def gen_quarterly_report(path: Path) -> None:
    """DOCX：表格密集（3 张表），验证 docx 表格块整块保留。"""
    import docx

    doc = docx.Document()
    doc.add_heading("2026 年度经营运营季度回顾", level=1)
    doc.add_paragraph("本报告汇总全年四个季度的营收、订单与成本结构数据，供管理层决策参考。")

    doc.add_heading("一、季度营收与毛利", level=2)
    t1 = doc.add_table(rows=5, cols=4)
    t1.style = "Table Grid"
    for r, row in enumerate([
        ("季度", "营收(万元)", "成本(万元)", "毛利率"),
        ("Q1", "4200", "3024", "28.0%"),
        ("Q2", "4680", "3276", "30.0%"),
        ("Q3", "5130", "3591", "30.0%"),
        ("Q4", "5720", "3947", "31.0%"),
    ]):
        for c, val in enumerate(row):
            t1.cell(r, c).text = val

    doc.add_heading("二、区域订单量", level=2)
    t2 = doc.add_table(rows=5, cols=3)
    t2.style = "Table Grid"
    for r, row in enumerate([
        ("区域", "订单量(单)", "同比增长"),
        ("华东", "12800", "15%"),
        ("华南", "9600", "9%"),
        ("华北", "7400", "-3%"),
        ("西部", "3100", "22%"),
    ]):
        for c, val in enumerate(row):
            t2.cell(r, c).text = val

    doc.add_heading("三、成本构成", level=2)
    t3 = doc.add_table(rows=5, cols=3)
    t3.style = "Table Grid"
    for r, row in enumerate([
        ("项目", "金额(万元)", "占比"),
        ("原材料", "8600", "62%"),
        ("人工", "2400", "17%"),
        ("物流", "1500", "11%"),
        ("其他", "1400", "10%"),
    ]):
        for c, val in enumerate(row):
            t3.cell(r, c).text = val

    doc.add_paragraph("全年累计营收 19730 万元，整体毛利率提升至 29.8%。")

    doc.save(str(path))


# ══════════════════════════════════════════════════════════════════════
# PDF 语料
# ══════════════════════════════════════════════════════════════════════

GUARDIAN_PAGES: list[tuple[str, list[str]]] = [
    (
        "Guardian X1 智能巡检终端 用户手册",
        [
            "第一章 产品概述\n"
            "Guardian X1 是一款面向工业场景的智能巡检终端，支持红外测温与三轴振动采集。\n"
            "设备防护等级为 IP67，工作温度范围为零下 20 摄氏度至 60 摄氏度。\n"
            "整机重量 480 克，配备 5.5 英寸电容触摸屏，支持戴手套操作。",
            "第二章 技术参数\n"
            "处理器为四核 1.8GHz，内存 4GB，存储 64GB。\n"
            "电池容量为 6000mAh，连续巡检续航约 12 小时。\n"
            "无线连接支持 Wi-Fi 6 与蓝牙 5.2，可选配 4G 模块。",
        ],
    ),
    (
        "Guardian X1 智能巡检终端 用户手册",
        [
            "第三章 使用说明\n"
            "开机后长按电源键 3 秒进入主界面。\n"
            "首次使用需完成设备绑定，绑定码为 8 位数字。\n"
            "巡检数据默认每小时自动同步一次，弱网环境下会自动重试三次。",
            "第四章 维护保养\n"
            "每 6 个月更换一次防尘滤网，粉尘工况下应缩短至每 3 个月。\n"
            "设备进水后应立即断电并送修，禁止自行拆解。\n"
            "长期存放时应保持电量在 50% 左右，并置于干燥环境。",
        ],
    ),
]

WAREHOUSE_PAGES: list[tuple[str, list[str]]] = [
    (
        "仓储作业规范 内部文件",
        [
            "第一章 收货作业\n"
            "到货后由收货员核对送货单与采购订单的一致性，核对内容包括品名、规格与数量。\n"
            "数量差异超过 2% 的，须在系统内发起异常单并通知采购人员跟进。\n"
            "收货完成后 2 小时内完成系统入库登记，禁止隔日补录。",
            "第二章 上架作业\n"
            "上架前须完成外观检查与条码校验，破损货物转入待处理区。\n"
            "上架库位遵循就近原则，重货下架、轻货上架，同批次货物尽量集中存放。",
        ],
    ),
    (
        "仓储作业规范 内部文件",
        [
            "第三章 拣货与复核\n"
            "拣货按波次进行，单个波次不超过 30 张订单。\n"
            "拣货完成后须经复核岗二次核对，复核通过率纳入班组考核。\n"
            "错拣率月度目标为不高于 0.3%。",
            "第四章 盘点与发货\n"
            "循环盘点每周执行一次，全盘每季度执行一次。\n"
            "盘点差异超过 500 元的须提交差异说明并由仓储经理审批。\n"
            "发货前核对承运商信息，出库单随货同行联须加盖出库章。",
        ],
    ),
]


def _write_pdf(path: Path, pages: list[tuple[str, list[str]]]) -> None:
    """逐页写 PDF：每页恒定页眉/页脚（跨页重复 → 验证页眉页脚剔除）。

    `insert_font(fontfile=...)` 会把中文字体**整包**嵌入（simhei ≈ 9.3MB），
    2 页文档就能撑到 9.7MB。故先落盘 staging，再用 `subset_fonts()` 只保留
    实际用到的字形重新保存——体积降到约 13KB，且经 `parse_file` 验证中文抽取
    与页眉页脚剔除均不受影响。缺 fontTools 时自动回退为未子集版本（不失败）。
    """
    import pymupdf

    font_file = _cjk_font()
    staging = path.with_name(path.name + ".stage")
    doc = pymupdf.open()
    for header, blocks in pages:
        page = doc.new_page()
        if font_file:
            page.insert_font(fontname="cjk", fontfile=font_file)
            fontname = "cjk"
        else:
            fontname = "helv"

        page.insert_textbox(pymupdf.Rect(50, 40, 545, 62), header,
                            fontsize=9, fontname=fontname)
        y = 80
        for block in blocks:
            page.insert_textbox(pymupdf.Rect(50, y, 545, y + 220), block,
                                fontsize=11, fontname=fontname)
            y += 230
        page.insert_textbox(pymupdf.Rect(50, 770, 545, 792), "锐眼科技 版权所有",
                            fontsize=9, fontname=fontname)
    doc.save(str(staging), garbage=4, deflate=True)
    doc.close()

    if font_file:
        try:
            with pymupdf.open(str(staging)) as sub:
                sub.subset_fonts()
                sub.save(str(path), garbage=4, deflate=True)
            staging.unlink()
            return
        except Exception as exc:  # noqa: BLE001 —— 缺 fontTools 等，回退不阻断
            print(f"  [warn] {path.name} 字体子集化失败，保留未子集版本：{exc}")
    staging.replace(path)


def gen_guardian_manual(path: Path) -> None:
    _write_pdf(path, GUARDIAN_PAGES)


def gen_warehouse_ops(path: Path) -> None:
    _write_pdf(path, WAREHOUSE_PAGES)


# ══════════════════════════════════════════════════════════════════════
# 编排
# ══════════════════════════════════════════════════════════════════════

# (文件名, 生成函数, 分组说明)
DOCS: list[tuple[str, Callable[[Path], None], str]] = [
    # ── 强干扰：差旅报销双版本（数值互相冲突）──
    ("policy_travel_expense_v1_2025-03-01.md", gen_travel_v1, "强干扰 差旅v1"),
    ("policy_travel_expense_v2_2026-01-01.md", gen_travel_v2, "强干扰 差旅v2"),
    # ── 强干扰：同名产品线手册（结构雷同、参数不同）──
    ("manual_guardian_x1.pdf", gen_guardian_manual, "强干扰 手册A"),
    ("manual_sentinel_s2.docx", gen_sentinel_manual, "强干扰 手册B"),
    # ── 强干扰：假期与审批语义撞车 ──
    ("hr_annual_leave_2026.md", gen_annual_leave, "强干扰 年假"),
    ("policy_procurement_payment.docx", gen_procurement_flow, "强干扰 采购"),
    # ── 弱干扰：真实知识库噪声 ──
    ("misc_canteen_menu.txt", gen_canteen, "弱干扰 食堂"),
    ("misc_team_building_notice.txt", gen_team_building, "弱干扰 团建"),
    ("misc_parking_guide.md", gen_parking, "弱干扰 停车"),
    ("misc_onboarding_checklist.md", gen_onboarding, "弱干扰 入职"),
    # ── 规模 / 格式补位 ──
    ("kb_long_service_handbook.md", gen_long_handbook, "长文档 多chunk"),
    ("report_quarterly_ops_2026.docx", gen_quarterly_report, "表格密集"),
    ("manual_warehouse_ops_2026-02-10.pdf", gen_warehouse_ops, "多页PDF"),
]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="扩充 M6 评估语料")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="输出目录（默认 data/samples）")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"输出目录：{out_dir}\n")
    total_bytes = 0
    for filename, gen, tag in DOCS:
        path = out_dir / filename
        gen(path)
        size = path.stat().st_size
        total_bytes += size
        print(f"  [ok] {filename:44} {size:>8,} B   {tag}")

    print(f"\n共生成 {len(DOCS)} 个文件，合计 {total_bytes:,} B")
    print("提示：下一步用项目真实解析链路验证——")
    print("  python scripts/verify_golden_anchors.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
