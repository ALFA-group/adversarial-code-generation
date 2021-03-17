echo "Datasets:"

if [ -d "/mnt/c2s/java-small" ] && [ -n "$(ls -A /mnt/c2s/java-small)" ]; then
  echo "  + c2s/java-small/..." 
  du -hs /mnt/c2s/java-small/* | while read -r line; do
    the_file=$(echo "${line}" | awk '{ print $2 }')
    the_size=$(echo "${line}" | awk '{ print $1 }')
    echo "    - '${the_file}' (${the_size})"
  done 
fi

if [ -d "/mnt/csn/java" ] && [ -n "$(ls -A /mnt/csn/java)" ]; then
  echo "  + csn/java/..." 
  du -hs /mnt/csn/java/* | while read -r line; do
    the_file=$(echo "${line}" | awk '{ print $2 }')
    the_size=$(echo "${line}" | awk '{ print $1 }')
    echo "    - '${the_file}' (${the_size})"
  done 
fi

if [ -d "/mnt/csn/python" ] && [ -n "$(ls -A /mnt/csn/python)" ]; then
  echo "  + csn/python/..." 
  du -hs /mnt/csn/python/* | while read -r line; do
    the_file=$(echo "${line}" | awk '{ print $2 }')
    the_size=$(echo "${line}" | awk '{ print $1 }')
    echo "    - '${the_file}' (${the_size})"
  done 
fi

if [ -d "/mnt/sri/py150" ] && [ -n "$(ls -A /mnt/sri/py150)" ]; then
  echo "  + sri/py150/..." 
  du -hs /mnt/sri/py150/* | while read -r line; do
    the_file=$(echo "${line}" | awk '{ print $2 }')
    the_size=$(echo "${line}" | awk '{ print $1 }')
    echo "    - '${the_file}' (${the_size})"
  done 
fi
