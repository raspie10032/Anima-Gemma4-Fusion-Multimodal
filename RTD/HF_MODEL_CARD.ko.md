---
language:
- en
- ko
tags:
- gemma
- gguf
- lora
- multimodal
- image-generation
- anime
- danbooru
- local-runtime
license: other
---

# GemmAnima 프로토타입 어댑터 번들

[English model card](README.md) | [GitHub app source](https://github.com/raspie10032/Anima-Gemma4-Fusion-Multimodal)

## 프로토타입 안내

이 저장소는 GemmAnima 독립 앱을 위한 **v0.1 공개 프로토타입 어댑터/체크포인트 번들**입니다. 현재 로컬 파이프라인을 재현하고 테스트할 수 있도록 공개한 것이며, safety-rated assistant, production image model, commercial-ready release가 아닙니다.

이 HF 저장소는 GemmAnima가 만든 어댑터, projector, bridge checkpoint, metadata, 문서만 포함합니다. 원본 base weight는 각 원본 모델 페이지에서 받아야 하며, 여기서 재배포하지 않습니다.

## 요약

| 항목 | 상태 |
| --- | --- |
| 릴리즈 유형 | 공개 프로토타입 어댑터/체크포인트 번들 |
| Base weights | 포함하지 않음 |
| 런타임 구조 | Gemma base GGUF 1개 + task LoRA/mmproj + Anima base weights + bridge profiles |
| 평가 상태 | 로컬 smoke test와 bridge check 중심 |
| Safety | safety-rated 아님 |
| 라이선스 | GemmAnima notice와 upstream base-model 제한을 함께 적용 |

GemmAnima는 Gemma/TIPO 계열 언어 및 비전 태깅 코어를 Anima 이미지 생성 코어에 연결하는 로컬 multimodal prototype입니다. HiddenStage Bridge가 Gemma 쪽 계획을 Anima conditioning으로 매핑합니다.

## 파일 구성

### Gemma Core

| 파일 | 역할 |
| --- | --- |
| `gemma_core/text-adapter-model-f16.gguf` | 텍스트/채팅 LoRA adapter |
| `gemma_core/vision-tagger-adapter-model-f16.gguf` | mixed-pose-front v2 final 기반 비전/태거 LoRA adapter |
| `gemma_core/gemma4-tipo-vision.mmproj-f16.gguf` | mixed-pose-front v2 final과 짝이 되는 vision projector |

외부 필요 파일:

| 파일 | 받는 곳 |
| --- | --- |
| `gemma-4-E2B-it-heretic-ara-custom.Q4_K_M.gguf` | `mradermacher/gemma-4-E2B-it-heretic-ara-custom-GGUF` 원본 페이지 |

권장 런타임 형태는 upstream base GGUF를 하나 로드하고 llama.cpp `--lora`로 GemmAnima task adapter를 붙이는 방식입니다. 여러 개의 full merged GGUF를 배포하는 방향이 아닙니다.

### Anima Image Core

Anima base weight는 이 저장소에 포함하지 않습니다.

| 파일 | 역할 |
| --- | --- |
| `split_files/diffusion_models/anima-base-v1.0.safetensors` | `circlestone-labs/Anima`에서 다운로드 |
| `split_files/vae/qwen_image_vae.safetensors` | `circlestone-labs/Anima`에서 다운로드 |

Anima text encoder weight는 현재 standalone runtime의 필수 파일이 아닙니다. 현재 in-process renderer는 conditioning shape 호환을 위한 tokenizer-format metadata를 사용합니다.

지원 generation control:

| 항목 | 값 |
| --- | --- |
| Sampler | `euler`, `euler_ancestral`, `dpmpp_2m`, `dpmpp_2m_sde_gpu` |
| Scheduler | `normal`, `karras`, `sgm_uniform` |
| Resolution presets | `1024x1024`, `832x1216`, `768x1344`, custom |

### HiddenStage Bridge

| 파일 | 역할 |
| --- | --- |
| `hiddenstage_bridge/hiddenstage-planner-adapter.safetensors` | planner LoRA adapter |
| `hiddenstage_bridge/hiddenstage-planner-embed-vision.pt` | planner vision embedding |
| `hiddenstage_bridge/kv_proj_hiddenstage_planner_v2.pt` | HiddenStage bridge checkpoint |
| `hiddenstage_bridge/kv_proj_balanced_pose_153k_pose10k_a0p35.pt` | 일반 이미지 및 pose-sensitive prompt용 기본 bridge profile |
| `hiddenstage_bridge/kv_proj_style_artist_v37a_10k.pt` | style tag 및 rare surface token용 bridge profile |
| `hiddenstage_bridge/kv_proj_text_exact_v27_alpha35.pt` | sign, label, caption, readable text prompt용 bridge profile |

이 bridge profile들은 프로토타입 routing choice이며, 별도 평가 없이 promoted model로 설명하면 안 됩니다.

## 사용 목적

이 번들은 다음 용도에 맞습니다.

- GemmAnima 앱 로컬 런타임 테스트
- 한국어/영어 채팅과 강한 언어 하네스
- 태그 요청 시 canonical English Danbooru tag 출력
- 채팅 기반 이미지 생성 요청 계획
- 앱이 관리하는 preset system을 통한 Anima 렌더링

한국어로 대화하더라도 태그 요청의 출력 태그는 영어 Danbooru canonical tag여야 합니다.

## 제외 범위

이 번들은 다음을 보장하지 않습니다.

- 일반 목적 safety-filtered assistant
- 완전히 평가된 public image-generation model
- Gemma, Anima, NVIDIA Cosmos, dataset 라이선스 override
- pose, anatomy, text rendering, prompt fidelity 보장
- 상업 사용 가능성

## 크기

| 파트 | 대략적인 업로드 크기 |
| --- | ---: |
| Gemma Core adapters/projector | 약 1.06 GB |
| HiddenStage Bridge | 약 0.40 GB |
| Anima Image Core base weights | 이 저장소에 없음 |
| 총 업로드 크기 | 약 1.46 GB |

전체 로컬 런타임은 외부 base weight까지 별도로 받으면 대략 9 GB대입니다. 정확한 크기는 선택한 base 파일과 호환 파일 포함 여부에 따라 달라집니다.

## 설치와 다운로드

앱 소스는 GitHub 저장소에서 받습니다.

```powershell
git clone https://github.com/raspie10032/Anima-Gemma4-Fusion-Multimodal.git
cd Anima-Gemma4-Fusion-Multimodal
.\GemmAnima.bat bootstrap
.\GemmAnima.bat download
.\GemmAnima.bat
```

다운로드 계획을 JSON으로 확인:

```powershell
python -m gemmanima.cli model-download-plan --json
python -m gemmanima.cli ensure-model-assets --json
```

## 라이선스와 주의

이 번들은 upstream base model을 재라이선스하지 않습니다. 다음 항목을 함께 확인해야 합니다.

- Gemma/GGUF 원본 모델 페이지와 라이선스
- Anima 원본 모델 페이지와 CircleStone Labs Non-Commercial License
- NVIDIA Open Model License 관련 조건
- GemmAnima adapter/checkpoint notice
- 학습 데이터 및 tag source 제한

현재 Anima 경로가 포함되므로, 별도 검토와 upstream permission이 없는 한 v0.1 번들은 **비상업, 비프로덕션, 제한적 테스트용 프로토타입**으로 취급하세요.

## 릴리즈 체크리스트

v0.1 공개 프로토타입 기준:

- Base weights는 포함하지 않음
- Adapter/checkpoint만 포함
- `LICENSE_NOTICES.md` 포함
- `adapter_manifest_v0.1.json`에 byte size와 SHA256 기록
- 한국어/영어 문서 제공

promotion 전에 필요한 것:

- 재현 가능한 inference smoke 공개
- safety/content-policy review
- 대표 출력 예시와 평가 근거

