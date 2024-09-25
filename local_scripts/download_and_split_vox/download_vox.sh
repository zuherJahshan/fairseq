# download the following links:
# https://dl.fbaipublicfiles.com/librilight/data/large.tar
# https://dl.fbaipublicfiles.com/librilight/data/medium.tar
# https://dl.fbaipublicfiles.com/librilight/data/small.tar
# These files are fairly big, so they might fail in the middle of the download, make sure to run them with wget -c and with a while true loop until they are downloaded
while true; do
    wget -c https://dl.fbaipublicfiles.com/librilight/data/large.tar
    if [ $? -eq 0 ]; then
        break
    fi
done

while true; do
    wget -c https://dl.fbaipublicfiles.com/librilight/data/medium.tar
    if [ $? -eq 0 ]; then
        break
    fi
done

while true; do
    wget -c https://dl.fbaipublicfiles.com/librilight/data/small.tar
    if [ $? -eq 0 ]; then # meaning last command was successful
        break
    fi
done
