for x in *.jpg; do
  touch "${x%.*}.txt"
  echo "$(cd "$(dirname "$1")"; pwd -P)/${x%.*}.jpg" >> negative_labels.sh
done

