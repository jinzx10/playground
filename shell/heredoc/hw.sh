#!/bin/bash

DAY=MON
cat > a.txt <<EOF
	good day $DAY
EOF

cat >> a.txt <<-EOF
	good day $DAY
EOF

cat >> a.txt <<"EOF"
	good day $DAY
EOF

cat >> a.txt <<-"EOF"
	good day $DAY
EOF


