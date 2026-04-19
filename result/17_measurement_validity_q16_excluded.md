# 17 Q16_7 제외 전후 측정타당성 비교

본 분석은 조직지원 인식 척도에서 Q16_7 문항 포함 여부가 측정타당성 결과에 미치는 영향을 점검하기 위한 강건성 검토이다.

## 원래 척도: Q16_1 ~ Q16_7

### AVE / CR
| 구성개념 | 문항 | AVE | CR |
| --- | --- | ---: | ---: |
| work_effect | Q7_1, Q7_2, Q7_3, Q7_4, Q7_5 | 0.604 | 0.884 |
| org_support | Q16_1, Q16_2, Q16_3, Q16_4, Q16_5, Q16_6, Q16_7 | 0.442 | 0.836 |
| strategic_expectation | Q20_2, Q20_3, Q20_4 | 0.471 | 0.719 |
| motivation_voluntary | Q9_3, Q9_4 | 0.706 | 0.827 |

### Fornell-Larcker
```text
                       work_effect  ...  motivation_voluntary
work_effect                  1.000  ...                 0.432
org_support                  0.380  ...                 0.287
strategic_expectation        0.456  ...                 0.274
motivation_voluntary         0.432  ...                 1.000

[4 rows x 4 columns]
```

| 구성개념 | sqrt(AVE) |
| --- | ---: |
| work_effect | 0.777 |
| org_support | 0.665 |
| strategic_expectation | 0.687 |
| motivation_voluntary | 0.840 |

### HTMT
| 구성개념 쌍 | HTMT |
| --- | ---: |
| work_effect - org_support | 0.475 |
| work_effect - strategic_expectation | 0.594 |
| work_effect - motivation_voluntary | 0.510 |
| org_support - strategic_expectation | 0.620 |
| org_support - motivation_voluntary | 0.388 |
| strategic_expectation - motivation_voluntary | 0.376 |

---

## 대안 척도: Q16_1 ~ Q16_6 (Q16_7 제외)

### AVE / CR
| 구성개념 | 문항 | AVE | CR |
| --- | --- | ---: | ---: |
| work_effect | Q7_1, Q7_2, Q7_3, Q7_4, Q7_5 | 0.604 | 0.884 |
| org_support | Q16_1, Q16_2, Q16_3, Q16_4, Q16_5, Q16_6 | 0.507 | 0.855 |
| strategic_expectation | Q20_2, Q20_3, Q20_4 | 0.471 | 0.719 |
| motivation_voluntary | Q9_3, Q9_4 | 0.706 | 0.827 |

### Fornell-Larcker
```text
                       work_effect  ...  motivation_voluntary
work_effect                  1.000  ...                 0.432
org_support                  0.323  ...                 0.211
strategic_expectation        0.456  ...                 0.274
motivation_voluntary         0.432  ...                 1.000

[4 rows x 4 columns]
```

| 구성개념 | sqrt(AVE) |
| --- | ---: |
| work_effect | 0.777 |
| org_support | 0.712 |
| strategic_expectation | 0.687 |
| motivation_voluntary | 0.840 |

### HTMT
| 구성개념 쌍 | HTMT |
| --- | ---: |
| work_effect - org_support | 0.393 |
| work_effect - strategic_expectation | 0.594 |
| work_effect - motivation_voluntary | 0.510 |
| org_support - strategic_expectation | 0.561 |
| org_support - motivation_voluntary | 0.274 |
| strategic_expectation - motivation_voluntary | 0.376 |

---

## 주요 해석

- Q16_7은 기존 EFA에서 다른 문항들보다 상대적으로 낮은 적재값을 보였으므로, 제외 버전의 측정타당성 결과를 추가 비교하였다.
- 제외 버전에서 AVE, CR, HTMT가 더 안정적이면 Q16_7 제외 결정은 내용타당성과 판별타당성 측면에서 더 설득력 있다.
- 본 비교 결과는 조직지원 척도의 특정 문항 선택이 본 연구의 실증결과를 과도하게 좌우하는지 여부를 점검하기 위한 보조 근거로 사용한다.
