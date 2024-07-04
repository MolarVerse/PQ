for i in $(find src/ tests/ include/ apps/ -regex ".*\.\(hpp\|cpp\|c\|h\)$"); do

    if ! grep -q Copyright $i; then

        echo "Adding license to $i"
        cat config/licenseHeader.txt $i >$i.new && mv $i.new $i

    fi

done

for i in $(find integration_tests/ | grep "\.py"); do

    if ! grep -q Copyright $i; then

        echo "Adding license to $i"
        cat config/licenseHeaderPython.txt $i >$i.new && mv $i.new $i

    fi

done
