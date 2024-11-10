#!/bin/zsh
# crontab: 01 20 * * 1-5 /Users/bryanpalmer/ABS/notebooks/ASX-daily-capture.sh
# note: * * * * * ==> minute hour day-of-month month day-of-week

# set-up parameters
home=/Users/bryanpalmer
working=ABS/notebooks
runrun=asx_daily_data_capture.py
mmenv=312

# move to the home directory
cd $home

# activate the micromamba environment
micromamba activate $mmenv

# move to the working directory
cd $working

# initiate the data capture
$home/micromamba/envs/$mmenv/bin/python ./$runrun >>./LOGS/asx-log.log 2>>./LOGS/asx-err.log

# update git
#git commit "../betting-data/sportsbet-2025-election-winner.csv" -m "data update"
#git push
