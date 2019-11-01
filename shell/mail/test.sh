#!/bin/bash

mail -s 'Test' -a ./attachment1.txt -a ./attachment2.txt zuxinjin@sas.upenn.edu < ./textfile.txt


#mail -s 'Test' zuxinjin@sas.upenn.edu jzx016@hotmail.com << EOF
#Dear Recipient,
#
#This is a shell-script test.
#Hello World!
#
#Zuxin
#EOF
