# AI Engineer Code Mentor

> ML 기초 지식이 있는 개발자를 위한 AI 코드 멘토 스킬

[![Claude Code](https://img.shields.io/badge/Claude%20Code-Skill-blue)](https://claude.ai/claude-code)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 소개

AI가 생성한 ML 코드를 **진짜 이해하고 싶은** 개발자를 위한 스킬입니다.

단순히 "이 코드는 데이터를 정규화합니다"가 아닌, **"왜 정규화하는지, 왜 이 숫자인지, 안 하면 어떻게 되는지"**를 설명합니다.

### 대상 청중

- ✅ Python 기초 지식 보유
- ✅ ML 용어(gradient, loss, optimizer 등) 알고 있음
- ✅ "왜 이렇게 했는지"가 궁금한 개발자
- ❌ ML 완전 초보자 (용어 설명이 필요한 경우)

## 주요 기능

### 6가지 분석 모드

| 모드 | 용도 | 예시 |
|------|------|------|
| `line-by-line` | 코드 라인별 해석 | `/aiengineer-code-mentor line-by-line model.py` |
| `deep` | 특정 함수/클래스 심층 분석 | `/aiengineer-code-mentor deep train.py:train_epoch` |
| `quiz` | 이해도 확인 퀴즈 | `/aiengineer-code-mentor quiz model.py` |
| `why` | 특정 라인 "왜?" 질문 | `/aiengineer-code-mentor why model.py:42` |
| `break-it` | "망하면?" 시나리오 | `/aiengineer-code-mentor break-it model.py` |
| `origins` | 코드 패턴 출처 추적 | `/aiengineer-code-mentor origins model.py` |

### 답변 가능한 질문 예시

```
Q: "max_depth=-1인데 왜 num_leaves로 제어해?"
A: LightGBM은 leaf-wise 방식이라 깊이보다 리프 수가 복잡도와 직결됩니다.
   num_leaves=31 = 2^5-1로, max_depth=5 수준의 복잡도를 리프 수로 표현한 것입니다.

Q: "colsample_bytree=0.8이 왜 트리 다양성 증가야?"
A: 각 트리가 80%의 다른 피처 조합을 보면서 학습하면,
   트리 간 상관관계가 낮아지고 앙상블 효과가 증가합니다.

Q: "num_leaves=31은 어디서 나온 숫자야?"
A: 31 = 2^5 - 1. LightGBM이 max_depth=5 수준을 기본으로 상정한 값입니다.
```

## 설치 방법

### Claude Code에서 사용

```bash
# 저장소 클론
git clone https://github.com/black940514/aiengineer-code-mentor.git

# Claude Code commands 폴더에 복사
cp -r aiengineer-code-mentor ~/.claude/commands/
```

### 또는 직접 다운로드

1. 이 저장소의 `SKILL.md`와 `references/` 폴더를 다운로드
2. `~/.claude/commands/aiengineer-code-mentor/`에 배치

## 사용법

### 기본 사용

```bash
# 전체 프로젝트 분석
/aiengineer-code-mentor

# 특정 파일 라인별 해석
/aiengineer-code-mentor line-by-line train.py

# 특정 라인 질문
/aiengineer-code-mentor why model.py:42
```

### 출력 파일

모든 분석 결과는 Markdown 파일로 저장됩니다:

| 모드 | 출력 파일 |
|------|----------|
| `line-by-line` | `{filename}_explained.md` |
| `deep` | `{function}_deep.md` |
| `quiz` | `{filename}_quiz.md` |
| `why` | `{filename}_why_L{line}.md` |
| `break-it` | `{filename}_breakit.md` |
| `origins` | `{filename}_origins.md` |

## 파일 구조

```
aiengineer-code-mentor/
├── SKILL.md                    # 메인 스킬 정의
└── references/
    ├── why-these-numbers.md    # 매직넘버 출처 (42, 0.8, 31, 1e-4 등)
    ├── algorithm-internals.md  # 알고리즘 내부 동작 (leaf-wise, Adam 등)
    ├── parameter-interactions.md # 파라미터 간 관계
    ├── cv-concepts.md          # 컴퓨터 비전 개념
    ├── ml-pipeline.md          # ML 파이프라인 개념
    └── timeseries-concepts.md  # 시계열 개념
```

## Reference 파일 설명

### why-these-numbers.md
ML 코드에서 자주 보이는 "마법의 숫자"들의 출처:
- `42` - 시드 고정 (은하수를 여행하는 히치하이커)
- `[0.485, 0.456, 0.406]` - ImageNet 정규화 평균값
- `num_leaves=31` - 2^5-1, LightGBM 기본값
- `learning_rate=1e-4` - Adam 논문 권장값

### algorithm-internals.md
알고리즘이 내부적으로 어떻게 작동하는지:
- Level-wise vs Leaf-wise 트리 성장
- `model.eval()` vs `torch.no_grad()` 차이
- Adam 옵티마이저 내부 동작
- BatchNorm vs LayerNorm

### parameter-interactions.md
함께 조정해야 하는 파라미터 쌍:
- `learning_rate` ↔ `n_estimators`
- `num_leaves` ↔ `max_depth`
- `batch_size` ↔ `learning_rate`
- `dropout` ↔ `weight_decay`

## 기여 방법

1. Fork this repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 라이선스

MIT License - 자유롭게 사용, 수정, 배포 가능

## 관련 링크

- [Claude Code 공식 문서](https://docs.anthropic.com/claude-code)
- [Anthropic Skills 저장소](https://github.com/anthropics/skills)
