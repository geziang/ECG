
## 2026-10-19 19:56 preflight FAIL: clocs s0 SHA c1f139d98059812aeee6453c5c7ec9cb2e99d57f338178833cd6e0cd159574d5 != 冻结 c1f139d98059812aeee6453c5c7ec9cb2e99d57f338178833cd4e0cd159574d5

## 2026-10-19 20:0x preflight 首轮 FAIL 定因=主机A自身誊写笔误(非任务书/非文件篡改)
- 现象: clocs s0 SHA 盘上 `...cd6e0cd...` vs runner 冻结表 `...cd4e0cd...`(单字符差),其余 17 项+三域 split SHA 全过。
- 取证: W4/baseline_results.csv(09-23 入档)clocs s0 四次在案(lp-cpsc/ft10-ptbxl/lp-ptbxl/lp-chapman)均=`...cd6e0cd...`;任务书§4.3 原文(git diff 证明未被改动)亦=`...cd6e0cd...`;clocs s2/s4 两侧一致。
- 结论: 笔误在 runner 的 SHA_FROZEN 常量誊写(6→4),任务书与 checkpoint 文件均正确。
- 处置: 修正 runner 常量;任务书保持原样零改动;曾误加的勘误注记已撤回。preflight 门按设计拦住了不一致,闭环。
