# AI Engineer Code Mentor

> AI/ML 기초 지식이 있는 개발자를 위한 코드 멘토 스킬 | **Computer Vision 엔지니어 특화**

[![Claude Code](https://img.shields.io/badge/Claude%20Code-Skill-blue)](https://claude.ai/claude-code)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Computer Vision](https://img.shields.io/badge/Focus-Computer%20Vision-orange)](https://github.com/black940514/aiengineer-code-mentor)

## 소개

AI가 생성한 코드를 **진짜 이해하고 싶은** AI 엔지니어를 위한 스킬입니다.

단순히 "이 코드는 데이터를 정규화합니다"가 아닌, **"왜 정규화하는지, 왜 이 숫자인지, 안 하면 어떻게 되는지"**를 설명합니다.

### 대상 청중

- ✅ **Computer Vision 엔지니어** (CNN, Object Detection, Segmentation 등)
- ✅ AI/ML 기초 지식 보유 (gradient, loss, optimizer 등)
- ✅ PyTorch/TensorFlow 사용 경험
- ✅ "왜 이렇게 했는지"가 궁금한 개발자
- ❌ AI/ML 완전 초보자 (용어 설명이 필요한 경우)

### 특화 분야

| 분야 | 다루는 내용 |
|------|-----------|
| **Computer Vision** | CNN, ResNet, YOLO, Segmentation, Augmentation, Transfer Learning |
| **Deep Learning** | 옵티마이저, 정규화, 학습률 스케줄링, Batch/Layer Norm |
| **ML Pipeline** | 데이터 분할, 손실 함수, 평가 지표, 과적합 방지 |
| **시계열** | LSTM/GRU, 슬라이딩 윈도우, 데이터 누수 방지 |

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

**Computer Vision 관련:**
```
Q: "왜 ImageNet 정규화 값이 [0.485, 0.456, 0.406]이야?"
A: ImageNet 120만장 이미지의 RGB 채널별 평균값입니다.
   Pretrained 모델 사용 시 반드시 동일 값을 써야 합니다.

Q: "왜 3x3 커널을 쓰는 거야? 5x5가 더 좋지 않아?"
A: VGGNet 논문에서 3x3 두 개 = 5x5 하나와 같은 receptive field지만,
   파라미터 수가 더 적다는 걸 발견했습니다. (18 vs 25)

Q: "model.eval() 안 하면 왜 문제야?"
A: Dropout이 추론 때도 작동해서 같은 입력에 다른 출력이 나옵니다.
   프로덕션에서 예측 불안정 → 디버깅 지옥.
```

**Deep Learning 관련:**
```
Q: "learning_rate=1e-4는 어디서 나온 숫자야?"
A: Adam 논문 권장값은 1e-3이지만, fine-tuning 시 pretrained 가중치를
   건드리지 않으려면 더 작은 1e-4~1e-5를 씁니다.

Q: "왜 BatchNorm인데 Transformer는 LayerNorm을 써?"
A: BatchNorm은 배치 크기에 의존하는데, Transformer는 시퀀스 길이가
   가변적이고 배치가 작을 수 있어서 LayerNorm이 더 적합합니다.
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

### cv-concepts.md ⭐ (Computer Vision 특화)
CV 엔지니어를 위한 핵심 개념:
- **이미지 표현**: 픽셀값, 채널, 축 순서 (NCHW vs NHWC)
- **데이터 증강**: RandomFlip, Rotation, ColorJitter 사용 시기
- **CNN 구조**: Conv, Pooling, kernel_size, stride, padding
- **전이 학습**: Feature extraction vs Fine-tuning 전략
- **Object Detection**: IoU, NMS, Two-stage vs One-stage

### why-these-numbers.md
AI 코드에서 자주 보이는 "마법의 숫자"들의 출처:
- `42` - 시드 고정 (은하수를 여행하는 히치하이커)
- `[0.485, 0.456, 0.406]` - ImageNet 정규화 평균값
- `3x3` - VGGNet에서 검증된 최적 커널 크기
- `learning_rate=1e-4` - Adam 논문 권장값
- `768` - BERT hidden size (12 heads × 64)

### algorithm-internals.md
알고리즘이 내부적으로 어떻게 작동하는지:
- `model.eval()` vs `torch.no_grad()` 차이
- Adam 옵티마이저 내부 동작
- BatchNorm vs LayerNorm
- Level-wise vs Leaf-wise 트리 성장

### parameter-interactions.md
함께 조정해야 하는 파라미터 쌍:
- `batch_size` ↔ `learning_rate` (Linear Scaling Rule)
- `dropout` ↔ `weight_decay` (정규화 중복 주의)
- `num_workers` ↔ `pin_memory` (DataLoader 최적화)
- `learning_rate` ↔ `n_estimators`

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
