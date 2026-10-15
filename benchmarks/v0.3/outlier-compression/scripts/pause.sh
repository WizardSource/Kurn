#!/bin/bash
# pause.sh stop|cont : SIGSTOP / SIGCONT the worker processes (llama-*, python) of the measurement queues
# (tmux sessions pipe, q2); their shells stay as they are and simply wait
sig=$([ "$1" = stop ] && echo STOP || echo CONT)
for s in pipe q2; do
  for p in $(tmux -f /exec-daemon/tmux.portal.conf list-panes -t $s -F '#{pane_pid}' 2>/dev/null); do
    sid=$(ps -o sid= -p $p | tr -d ' ')
    [ -n "$sid" ] && for q in $(ps -e -o pid=,sid=,comm= | awk -v s=$sid '$2 == s && $3 !~ /bash|sleep/ {print $1}'); do kill -$sig $q; done
  done
done
