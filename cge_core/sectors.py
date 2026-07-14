# -*- coding: utf-8 -*-
"""
国家统计局42部门分类标准 (NBS 2017)
"""

SECTORS_42 = [
    ("S01", "农林牧渔产品和服务",     "Agriculture, Forestry, Animal Husbandry, Fishery"),
    ("S02", "煤炭开采和洗选产品",     "Coal Mining and Washing"),
    ("S03", "石油和天然气开采产品",   "Oil and Natural Gas Extraction"),
    ("S04", "金属矿采选产品",         "Metal Ore Mining"),
    ("S05", "非金属矿和其他矿采选产品", "Non-metallic and Other Mineral Mining"),
    ("S06", "食品和烟草",             "Food and Tobacco Products"),
    ("S07", "纺织品",                 "Textiles"),
    ("S08", "纺织服装鞋帽皮革羽绒",   "Textile Wearing Apparel, Footwear, Leather"),
    ("S09", "木材加工品和家具",       "Wood Processing and Furniture"),
    ("S10", "造纸印刷和文教体育用品", "Paper, Printing, Stationery, Sports Goods"),
    ("S11", "石油、炼焦产品和核燃料", "Petroleum, Coking, Nuclear Fuel Processing"),
    ("S12", "化学产品",               "Chemical Products"),
    ("S13", "非金属矿物制品",         "Non-metallic Mineral Products"),
    ("S14", "金属冶炼和压延加工品",   "Metal Smelting and Rolling"),
    ("S15", "金属制品",               "Metal Products"),
    ("S16", "通用设备",               "General-Purpose Machinery"),
    ("S17", "专用设备",               "Special-Purpose Machinery"),
    ("S18", "交通运输设备",           "Transportation Equipment"),
    ("S19", "电气机械和器材",         "Electrical Machinery and Equipment"),
    ("S20", "通信设备、计算机和电子设备", "Communication, Computer, Electronic Equipment"),
    ("S21", "仪器仪表",               "Instruments and Meters"),
    ("S22", "其他制造产品和废品废料", "Other Manufacturing and Scrap"),
    ("S23", "金属制品、机械和设备修理服务", "Repair of Metal Products, Machinery, Equipment"),
    ("S24", "电力、热力生产和供应",   "Electricity, Heat Production and Supply"),
    ("S25", "燃气生产和供应",         "Gas Production and Supply"),
    ("S26", "水的生产和供应",         "Water Production and Supply"),
    ("S27", "建筑",                   "Construction"),
    ("S28", "批发和零售",             "Wholesale and Retail Trade"),
    ("S29", "交通运输、仓储和邮政",   "Transport, Storage, Postal"),
    ("S30", "住宿和餐饮",             "Accommodation and Food Services"),
    ("S31", "信息传输、软件和IT服务", "Information Transmission, Software, IT Services"),
    ("S32", "金融",                   "Finance"),
    ("S33", "房地产",                 "Real Estate"),
    ("S34", "租赁和商务服务",         "Leasing and Business Services"),
    ("S35", "科学研究和技术服务",     "Scientific Research and Technical Services"),
    ("S36", "水利、环境和公共设施管理", "Water, Environment, Public Facilities"),
    ("S37", "居民服务、修理和其他服务", "Resident Services, Repair, Other Services"),
    ("S38", "教育",                   "Education"),
    ("S39", "卫生和社会工作",         "Health and Social Work"),
    ("S40", "文化、体育和娱乐",       "Culture, Sports, Entertainment"),
    ("S41", "公共管理、社会保障和社会组织", "Public Administration, Social Security"),
    ("S42", "国际组织",               "International Organizations"),
]

# 便捷访问
SECTOR_CODES = [s[0] for s in SECTORS_42]
SECTOR_NAMES_CN = [s[1] for s in SECTORS_42]
SECTOR_NAMES_EN = [s[2] for s in SECTORS_42]
SECTOR_NAME_MAP = {s[0]: s[1] for s in SECTORS_42}
SECTOR_CODE_BY_NAME = {s[1]: s[0] for s in SECTORS_42}
NUM_SECTORS = 42
