# GemmAnima Engine

[English README](README.md) | [Hugging Face adapter bundle](https://huggingface.co/raspie/gemmanima-adapter-bundle)

GemmAnima는 로컬 우선(local-first) 채팅 및 이미지 생성 오케스트레이션 앱의 v0.1 공개 프로토타입입니다. 목표는 하나의 독립 앱 안에서 대화, 태그 계획, 이미지 요청 판단, Anima 기반 렌더링까지 이어지는 흐름을 테스트할 수 있게 만드는 것입니다.

이 저장소는 앱 소스, 설정, 스키마, 테스트, 학습/평가 보조 스크립트를 담습니다. 대형 모델 가중치는 GitHub에 포함하지 않습니다. 첫 실행 시 앱이 원본 모델 페이지와 GemmAnima Hugging Face 어댑터 번들에서 필요한 파일을 내려받도록 설계되어 있습니다.

## 핵심 구조

| 파트 | 역할 |
| --- | --- |
| Gemma Core | 상주 채팅, 언어 하네스, 의도 라우팅, Danbooru 태그 출력, 비전 태깅 |
| Anima Image Core | 로컬 이미지 생성과 VAE 디코딩 |
| HiddenStage Bridge | Gemma 쪽 계획/hidden-state를 Anima conditioning으로 매핑 |

기본 방향은 같은 Gemma 베이스 모델을 여러 번 복제하지 않고, 하나의 base GGUF에 LoRA/mmproj/bridge 파일을 붙여 쓰는 것입니다.

## 현재 기능

- 로컬 웹 GUI 기반 채팅 및 이미지 요청
- Gemma 런타임 상주 설계
- 한국어/영어 대화 언어 하네스
- 사용자가 한국어로 말해도 태그 요청은 canonical English Danbooru tag로 출력
- 일반 채팅, 이미지 생성, 이미지 태깅 자동 라우팅
- 채팅창 이미지 첨부 및 드래그 앤 드롭
- 이미지 생성 중 로딩 상태와 생성 단계 표시
- 해상도, sampler, scheduler, step, CFG, seed, Anima LoRA 사용 여부 프리셋
- 모델 다운로드 계획 및 진행률 API
- dry-run, local-worker, in-process, external-script 렌더러 모드
- 라우팅, 다운로드 계획, manifest, GUI 표면에 대한 테스트

## 설치

Windows에서는 `GemmAnima.bat` 하나만 사용자 실행 파일로 사용합니다.

```powershell
git clone https://github.com/raspie10032/Anima-Gemma4-Fusion-Multimodal.git
cd Anima-Gemma4-Fusion-Multimodal
.\GemmAnima.bat bootstrap
.\GemmAnima.bat health
```

필요 모델을 확인하고 다운로드하려면:

```powershell
.\GemmAnima.bat download
```

GUI 실행:

```powershell
.\GemmAnima.bat
```

기본 주소:

```text
http://127.0.0.1:8765
```

## 모델 파일 정책

GitHub 저장소는 소스와 설정만 관리합니다. 모델은 다음 원칙으로 받습니다.

| 파일군 | 다운로드 위치 |
| --- | --- |
| Gemma base GGUF | 원본 GGUF 모델 페이지 |
| Anima diffusion/VAE | 원본 Anima 모델 페이지 |
| GemmAnima LoRA/mmproj/bridge | GemmAnima Hugging Face adapter bundle |

정확한 다운로드 계획:

```powershell
python -m gemmanima.cli model-download-plan --json
```

## Hugging Face 번들

어댑터와 bridge 체크포인트는 다음 HF 저장소에 있습니다.

- [raspie/gemmanima-adapter-bundle](https://huggingface.co/raspie/gemmanima-adapter-bundle)
- 한국어 모델카드: `HF_MODEL_CARD.ko.md`

HF 번들은 base weight를 재배포하지 않습니다. 원본 모델의 라이선스와 제한을 그대로 따라야 합니다.

## 라이선스와 제한

GemmAnima는 Gemma, Anima, NVIDIA 관련 라이선스, GemmAnima 어댑터/bridge 파일, 학습 데이터 제한이 함께 걸리는 복합 프로토타입입니다. 현재 v0.1은 production-ready나 safety-rated 모델이 아닙니다.

Anima 경로가 포함되므로, 별도의 라이선스 검토와 필요한 upstream permission이 없는 한 **비상업, 비프로덕션, 제한적 테스트용 프로토타입**으로 취급하세요.

자세한 내용:

- `LICENSE.md`
- `RTD/LICENSE_NOTICES.md`
- `RTD/HF_MODEL_CARD.md`
- `RTD/HF_MODEL_CARD.ko.md`

## 개발자 메모

- 생성 이미지, run manifest, 캐시, 다운로드 모델, 체크포인트는 커밋하지 않습니다.
- RTX 5060은 호환 PyTorch 빌드가 준비되기 전까지 training/cache 작업에서 제외하는 것이 기본 전제입니다.
- 모델을 promote하거나 공개 주장에 넣기 전에는 반드시 별도 평가 결과가 필요합니다.

