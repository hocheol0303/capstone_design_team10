for d in */; do
  count=$(find "$d" -maxdepth 1 -type f | wc -l)
  [ "$count" -ne 4 ] && echo "$d: $count개 파일 ❌"
done