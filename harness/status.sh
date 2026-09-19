#!/bin/bash
# bio-auditor stage 1 status. Run:  ~/bixbench/status.sh
cd ~/bixbench
TOTAL=122
done_n=$(ls data/trajectories/bioauditor-s1/*.json 2>/dev/null | wc -l)
log=$(ls -t run_full*.log 2>/dev/null | head -1)
alive=$(pgrep -f run_bioauditor.py >/dev/null && echo RUNNING || echo "NOT RUNNING")

echo "=============================================="
echo " bio-auditor stage 1   [$alive]   $(date +%H:%M:%S)"
echo "=============================================="
printf " trajectories : %d / %d  (%d%%)\n" "$done_n" "$TOTAL" "$((done_n*100/TOTAL))"
printf " containers   : %s / 4 busy\n" "$(docker ps -q | wc -l)"
printf " errors       : %s traceback(s), %s timeout(s)\n" \
  "$(grep -ci traceback $log 2>/dev/null)" "$(grep -ci 'timeout occurred' $log 2>/dev/null)"
printf " disk free    : %s on C:\n" "$(df -h /mnt/c | tail -1 | awk '{print $4}')"
echo
echo " recent completions:"
ls -lt --time-style=+%H:%M:%S data/trajectories/bioauditor-s1/*.json 2>/dev/null | head -5 |
  awk '{n=split($7,p,"/"); printf "   %s  %s\n", $6, p[n]}'
echo
echo " in flight now:"
docker ps --format '   {{.Status}}' | head -4
echo
# rate + ETA from the last hour of completions
recent=$(find data/trajectories/bioauditor-s1 -name '*.json' -mmin -60 | wc -l)
if [ "$recent" -gt 0 ]; then
  left=$((TOTAL-done_n))
  printf " rate         : %s/hour   ETA ~%.1f h for the remaining %s\n" \
    "$recent" "$(echo "$left $recent" | awk '{printf "%.1f", $1/$2}')" "$left"
fi
echo " unanswered   : $(grep -l '"agent_answer": null' data/trajectories/bioauditor-s1/*.json 2>/dev/null | wc -l)"
echo
echo " live log tail:  tail -f ~/bixbench/$log"
