# 02 Preprocessing Summary

Built the analysis-ready dataset and checked the main control variables.

## Dataset output

- Output file: `chatbot_output_selected_preprocessed.csv`
- Total respondents: 1608
- AI users: 377
- AI non-users: 1231

## Preview of key derived variables
```text
   gender  rank_code  career_code  ai_task_count
0       0          3            4              0
1       0          5            4              0
2       1          3            3              0
3       1          1            1              0
4       0          1            1              1
```

## Overall sample characteristics

====================================
gender distribution (N=1608)
====================================
          N     %
gender           
0       807  50.2
1       801  49.8

====================================
rank distribution (N=1608)
====================================
             N     %
rank_code           
1          254  15.8
2          304  18.9
3          487  30.3
4          359  22.3
5          165  10.3
6           33   2.1
7            6   0.4

====================================
career distribution (N=1608)
====================================
               N     %
career_code           
1            606  37.7
2            289  18.0
3            424  26.4
4            221  13.7
5             68   4.2

====================================
age_raw distribution (N=1608)
====================================
          N     %
SQ1              
20대     211  13.1
30대     601  37.4
40대     500  31.1
50대     287  17.8
60대 이상    9   0.6

====================================
organization_raw distribution (N=1608)
====================================
            N     %
SQ4                
광역지방자치단체  416  25.9
기초지방자치단체  647  40.2
중앙행정기관    545  33.9

## Key note
- `ai_task_count` is computed as the number of checked Q4 task-use categories.
- The preprocessed dataset is now ready for all downstream scripts.
