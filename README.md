# Transformer Encoder-Decoder를 이용한 한국어 형태소 분석 모델
### [2024 2학기 자연어처리 과제5]

### 개발 기간
> 2024.12.19 ~ 2024.12.25

### 개발 환경
> Python 3.12.6 (venv)<br>
> Pytorch 2.4.1 + CUDA 12.4<br>
> GTX1060 6GB Desktop<br>

### 설명
+ 동기
    + 자연어처리 수업 과제
+ 기획
    + Transformer encoder-decoder를 이용하여 시스템을 개발한다.
    + 학습 데이터는 '21세기 세종계획' 프로젝트로 구축된 말뭉치를 사용한다. (언어정보나눔터에서 다운로드 가능)
    + 훈련:테스트는 95:5 비율로 나눈다.
    + 배치 번호를 입력받아 형태소 분석 결과를 출력

#### 옵티마이저 및 하이퍼파라미터
> optimizer = AdamW <br>
> learning rate = 0.000005 <br>
> epoch = 1 <br>
> 배치 크기 = 16 <br>
> 인코더 최대 토큰 길이 = 128 <br>
> 디코더 최대 토큰 길이 = 150 <br>

#### 학습 과정 및 성능지표
1 에포크만 수행하였는데, 이때 걸린시간은 약 4일이었으며 테스트 결과로 재현율이 약 93%로 계산되었다.
<img width="643" height="249" alt="noname01" src="https://github.com/user-attachments/assets/b765fb80-1818-43e2-94a8-1b864e9d127e" />
<img width="644" height="227" alt="noname02" src="https://github.com/user-attachments/assets/07038de9-2764-46b3-a84b-f52efc02088c" />

#### 입력 결과
<img width="643" height="59" alt="noname01" src="https://github.com/user-attachments/assets/c14460e6-9457-41e6-bf3c-70703fb98d24" />
<img width="643" height="60" alt="noname02" src="https://github.com/user-attachments/assets/5d6452b6-39a5-4fa3-86ec-47f1129d0e1d" />
<img width="643" height="79" alt="noname03" src="https://github.com/user-attachments/assets/5d3bbb05-ec1f-443d-a9b7-67489e936777" />
<img width="643" height="80" alt="noname04" src="https://github.com/user-attachments/assets/025a8d48-c175-4bd1-997f-b9c094e5f4ec" />
<img width="643" height="99" alt="noname05" src="https://github.com/user-attachments/assets/c2780489-3d8b-4f0b-8a03-7cb4fbd85ff7" />
<img width="643" height="81" alt="noname06" src="https://github.com/user-attachments/assets/2b1796e2-0c18-4fbe-89cb-7ded9a06283f" />
<img width="643" height="80" alt="noname07" src="https://github.com/user-attachments/assets/ef885187-02ae-494c-bfcd-9659c61530fe" />
<img width="643" height="80" alt="noname08" src="https://github.com/user-attachments/assets/d0dfd301-e0ab-44c5-98aa-f53684025327" />
<img width="643" height="62" alt="noname09" src="https://github.com/user-attachments/assets/982f254e-319e-4e32-a22d-0898bf7e5c84" />
<img width="643" height="60" alt="noname10" src="https://github.com/user-attachments/assets/80d14409-984e-466b-b91d-4a825c8ae9c7" />

<br>

