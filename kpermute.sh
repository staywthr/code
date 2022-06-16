#/bin/bash
#!/bin/bash
while IFS= read -r line; do
    python lme_label_kPermute.py $line
done < klist.txt