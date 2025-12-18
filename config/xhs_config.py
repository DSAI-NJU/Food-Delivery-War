# -*- coding: utf-8 -*-
# Copyright (c) 2025 relakkes@gmail.com
#
# This file is part of MediaCrawler project.
# Repository: https://github.com/NanmiCoder/MediaCrawler/blob/main/config/xhs_config.py
# GitHub: https://github.com/NanmiCoder
# Licensed under NON-COMMERCIAL LEARNING LICENSE 1.1
#

# 声明：本代码仅供学习和研究目的使用。使用者应遵守以下原则：
# 1. 不得用于任何商业用途。
# 2. 使用时应遵守目标平台的使用条款和robots.txt规则。
# 3. 不得进行大规模爬取或对平台造成运营干扰。
# 4. 应合理控制请求频率，避免给目标平台带来不必要的负担。
# 5. 不得用于任何非法或不当的用途。
#
# 详细许可条款请参阅项目根目录下的LICENSE文件。
# 使用本代码即表示您同意遵守上述原则和LICENSE中的所有条款。


# 小红书平台配置

# 排序方式，具体的枚举值在media_platform/xhs/field.py中
SORT_TYPE = "popularity_descending"

# 指定笔记URL列表, 必须要携带xsec_token参数
XHS_SPECIFIED_NOTE_URL_LIST = [
    "https://www.xiaohongshu.com/explore/687235fc0000000012020cbf?xsec_token=AB93L9IzGIIdobktipGQlQd_VzIAevN8dDaAshEyHDiO4=&xsec_source=pc_search",
    "https://www.xiaohongshu.com/explore/681a11d00000000021009c4e?xsec_token=ABxSW_1NZ24lu9RvtLOgI-HxTnltH0dBWzFtqeX8U68vE=&xsec_source=pc_search",
    "https://www.xiaohongshu.com/explore/692853bb000000001e016054?xsec_token=ABKEogoZZPfoM2ofIIbvKJHIMNik4MHVDUwb_jAeGcdQY=&xsec_source=pc_search",
    "https://www.xiaohongshu.com/explore/6885d60c0000000010012667?xsec_token=ABja2OI4cWiY_go18kHBvMen3KMzMJSsm8vz2UqZLHqMI=&xsec_source=pc_search",
    "https://www.xiaohongshu.com/explore/691729da000000001b021a5f?xsec_token=ABC_LHi985LKSR1pIpV-r38Hutc5ItLJk9g2hWwblGJVU=&xsec_source=pc_search",
    "https://www.xiaohongshu.com/explore/68130d350000000021004d04?xsec_token=ABVGknoYneQrRl-V2S_Jiq2aHZy7KBp2Ynd-2GC1XvS7g=&xsec_source=pc_search",
    "https://www.xiaohongshu.com/explore/68b30c1a000000001c006a59?xsec_token=ABsHAiiBidOtjiL7EEaxRosSSqyMiIoJWQOT5NvVEViTc=&xsec_source=pc_search",
    "https://www.xiaohongshu.com/explore/692aedd8000000001e002185?xsec_token=AB0XtmEU0SqUgIZHuYdPofiyuFcLsWtWG6jQvt_63wFSI=&xsec_source=pc_search",
    "https://www.xiaohongshu.com/explore/691c5549000000000d03d33e?xsec_token=ABywR_k3wriRn81Unekqj6dnkSwRqkwyoqu_IkqOWvZRw=&xsec_source=pc_search",
    "https://www.xiaohongshu.com/explore/686a1b190000000011003600?xsec_token=AB6Fj7I3Sm6s5DJneoLcPPaDdPCsTtc5bgy6y7bERFN_w=&xsec_source=pc_search",
    "https://www.xiaohongshu.com/explore/68b1aacf000000001c013b9e?xsec_token=ABGPop0dxqDO2z8nN_Hlb2gDZXQujOWjUNipzPf5dlWts=&xsec_source=pc_search",
    "https://www.xiaohongshu.com/explore/689d39aa000000001d024e81?xsec_token=ABE9V5hTMD9kBlKKFjSjTAm1c3jwNAi7nFM3DL1cbZbV8=&xsec_source=pc_search",
    "https://www.xiaohongshu.com/explore/6918287a000000000d00dde1?xsec_token=ABhIXuZXu6gG7Jv7a3fvCUs2l2JWC1ACE9s09fIAt7tv8=&xsec_source=pc_search",
    "https://www.xiaohongshu.com/explore/688c60c7000000002302f620?xsec_token=ABi6U-SzAWPYNa_TzkHemeiILH7bW1HQvz1BQtFkFHF18=&xsec_source=pc_search",
    "https://www.xiaohongshu.com/explore/692cf2ad000000001e0242b3?xsec_token=ABBKeeG8GVKDmd61xn6HPyDQNQtwaxcGsNGFZ7XpRfqMc=&xsec_source=pc_search",
    "https://www.xiaohongshu.com/explore/68f606c9000000000701711b?xsec_token=ABDA523Pq09mN9fCh5rk7hMQCl4EwzWUuGCXs-NMPCUNU=&xsec_source=pc_search",
    "https://www.xiaohongshu.com/explore/6929b268000000001e02110d?xsec_token=ABRv6DixBc-mhpoIOyMpbHEkrz47gniiKn86-SR1_oVAo=&xsec_source=pc_search",
    "https://www.xiaohongshu.com/explore/68d237a7000000001201cd29?xsec_token=ABOY-OsFgmusBij51fBrEqPb_C6fx2eMTPjsIAGrl31mc=&xsec_source=pc_search",
    "https://www.xiaohongshu.com/explore/68dc857b00000000040060d0?xsec_token=ABMjiBVK1SSGmYLlNmvbpQ94Xfkd9bW4PBe9itYJ_ASBY=&xsec_source=pc_search"
    # ........................
]

# 指定创作者URL列表，需要携带xsec_token和xsec_source参数

XHS_CREATOR_ID_LIST = [
    "https://www.xiaohongshu.com/user/profile/5f58bd990000000001003753?xsec_token=ABYVg1evluJZZzpMX-VWzchxQ1qSNVW3r-jOEnKqMcgZw=&xsec_source=pc_search"
    # ........................
]
