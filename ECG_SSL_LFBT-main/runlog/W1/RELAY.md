# RELAY.md — b0fast 收官接力操作单(2026-09-20 晚,主会话预授权)

> 触发条件:b0fast(main 树 run_pt,原 pid 57048)结束。任何会话(巡检/新主会话)按序执行;
> 冲突即中止,绝不强推。C1 NFH 链(worktree)与此流程互不阻塞。

1. **判死三重确认**:CIM 查 run_pt.py(排除 pt_pretrain_nfh 的 C1)+ Get-Process -Id 57048 直查 + GPU 显存交叉。三者一致才算结束。
2. **合并**(仅当 `ECG_SSL_LFBT-main/models/acl_region.py` 不存在):
   ```
   cd /f/新实验 && git merge w1-code-tasks --no-edit
   # 冲突(git status 出现 both modified)→ git merge --abort + 告警待人工,到此为止
   cd ECG_SSL_LFBT-main
   C:/Users/admin/.conda/envs/DL/python.exe -m py_compile run_pt.py models/acl_region.py   # 必须通过
   grep -c "acl_region" run_pt.py   # 必须 ≥2
   cd /f/新实验 && git commit -m "w1-code-tasks 合并: T1 D1L-fix + T2 ACL 四区(A1/A2/A3 解锁)"
   git -c http.lowSpeedLimit=0 -c http.lowSpeedTime=999 push -q origin main   # 失败=断网,注明待重试
   ```
3. **重启双车道**(仅当无 laneF/laneG2 runner 存活):
   ```
   powershell -NoProfile -Command "Start-Process -FilePath 'C:/Users/admin/.conda/envs/DL/python.exe' -ArgumentList 'pipeline_runner.py','--lane','F' -WorkingDirectory 'F:\新实验\ECG_SSL_LFBT-main' -WindowStyle Hidden -RedirectStandardOutput 'F:\新实验\ECG_SSL_LFBT-main\runlog\M\laneF.out.log' -RedirectStandardError 'F:\新实验\ECG_SSL_LFBT-main\runlog\M\laneF.err.log'"
   # 同样再起 --lane G2 → laneG2.out/err.log
   ```
   队列首项 a1_d1l_fix / a2_acl_intra 自动领取;C1 未完时第二车道候闸属正常。
4. **5 分钟后验证**:laneF.out.log 出现"启动任务 a1_d1l_fix_seed0"或"等待显存";pt_a1_d1l_fix_seed0.log 无 "unrecognized arguments"(出现=w1 合并有误,告警勿重试)。
5. **红线**:绝不杀进程;不改代码(合并是唯一 git 写操作);不动 worktree;不动 mimic。
6. **完成后**:b0fast 与 A 线读数均由 runner 自动落 csv;巡检每 2.5 小时照常报告;本单任务完成后作废。
