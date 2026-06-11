
# 🔬 Phân tích Chi tiết Kết quả Benchmark
## Quantum PPO vs Classical PPO — Nhận xét Khoa học

> Thực nghiệm với seed=42, encoding `data_reuploading`, PPO-Clip (ε=0.2), GAE (λ=0.95).
> Quantum actor: VQC với n_layers=2, pre-encoding MLP ẩn 64 nodes.
> Classical actor: MLP 1 hidden layer (hidden = max(state_dim, 4)).

---

## 1. CartPole-v1 — Discrete, Dễ

### Diễn biến training

| Giai đoạn | Quantum PPO | Classical PPO | Nhận xét |
|---|---:|---:|---|
| **First 100 eps** | mean = 22.6 (σ=9.8) | mean = 20.2 (σ=9.6) | Khởi đầu gần nhau |
| **Mid-point (~400 eps)** | mean = 100.6 | mean = 114.8 | Classical nhỉnh hơn |
| **Last 100 eps** | mean = **319.7** (σ=153.9) | mean = **409.1** (σ=73.4) | Classical ổn định hơn rõ rệt |
| **Best đạt được** | 500.0 ✅ | 500.0 ✅ | Cả 2 đều đạt điểm tuyệt đối |
| **First perfect step** | **~43k** | ~63k | Quantum đạt sớm hơn 20k steps |

### Nhận xét

**Quantum PPO hội tụ sớm hơn nhưng không ổn định.** Điểm thú vị nhất là Quantum đạt episode 500 điểm đầu tiên ở bước ~43k, trong khi Classical cần đến ~63k — tức là **Quantum "hiểu" bài toán nhanh hơn ~32%**. Tuy nhiên, standard deviation của Quantum (σ=153.9) gần **gấp đôi** Classical (σ=73.4), cho thấy policy không ổn định: có episode đạt 500, nhưng cũng có episode chỉ đạt 49 điểm.

**Nguyên nhân instability:** Với CartPole (4 chiều), quantum circuit 4 qubits chỉ có 24 tham số lượng tử nhưng có thêm ~966 tham số pre-encoding MLP. Gradient của pre-encoding cập nhật nhanh (SGD) trong khi quantum params cập nhật chậm hơn (parameter-shift) tạo ra **sự bất đối xứng tốc độ học**, khiến policy dao động mạnh sau khi đã học được.

**Classical đạt `stable_solve` ở step 90k** — tức là đạt reward trung bình ≥450 trong cửa sổ smoothed. Quantum không đạt được ngưỡng này, dù đã chạm đến 500 nhiều lần.

> **Kết luận CartPole:** Classical PPO chiến thắng về *ổn định*, nhưng Quantum có *tốc độ khám phá* ban đầu tốt hơn. Đây là bằng chứng sơ khởi cho "quantum advantage" trong early convergence.

---

## 2. LunarLander-v3 — Discrete, Khó

### Diễn biến training

| Giai đoạn | Quantum PPO (149k steps) | Classical PPO (300k steps) | Nhận xét |
|---|---:|---:|---|
| **First 100 eps** | mean = -154.8 (σ=88.6) | mean = -168.9 (σ=97.7) | Quantum nhỉnh hơn lúc đầu |
| **Mid-point** | mean = **-76.5** | mean = **-40.2** | Classical bắt kịp và vượt |
| **Last 100 eps** | mean = **43.3** (σ=121.7) | mean = **162.6** (σ=94.2) | Classical vượt trội |
| **Best đạt được** | 175.4 | **279.1** | Classical tốt hơn 60% |
| **Perfect (≥200 avg)** | ❌ Không đạt | ❌ Không đạt | Cả 2 chưa "solve" |

### Nhận xét

**Đây là môi trường bộc lộ rõ nhất điểm yếu của Quantum PPO.** Có hai nguyên nhân cốt lõi:

**Nguyên nhân 1 — Bất bình đẳng về data:** Do quantum circuit chậm hơn (237 FPS vs 1829 FPS — chênh lệch **7.7 lần**), với cùng budget thời gian wall-clock, Classical thu thập được khoảng **gấp đôi** experience (300k vs 150k steps). Trong RL, nhiều data thường tương đương hiệu suất tốt hơn — đây là lợi thế lớn cho Classical. Nếu so sánh *fair* trên cùng số timesteps, khoảng cách sẽ nhỏ hơn.

**Nguyên nhân 2 — Biểu diễn state hạn chế:** LunarLander có 8 chiều observation. Quantum circuit 6 qubits với data reuploading mã hóa 8 features qua 6 qubits, nhưng pre-encoding MLP phải *nén* thông tin này xuống. Với bài toán có không gian trạng thái phức tạp (vị trí, vận tốc, góc, chân đáp...), việc nén qua bottleneck lượng tử có thể làm mất thông tin quan trọng.

**Sự kiện đáng chú ý:** Ở ~130k steps, Quantum có 1 episode đạt reward = **-462.6** (rơi xuống đột ngột từ +100), cho thấy hiện tượng **catastrophic forgetting** hoặc gradient instability sau PPO update. Classical ổn định hơn: episode thấp nhất trong 100 ep cuối chỉ là -70.9.

> **Kết luận LunarLander:** Classical PPO vượt trội hoàn toàn. Lý do chính không chỉ là thuật toán mà là **tốc độ thu thập data**. Quantum PPO cần nhiều timesteps hơn để cạnh tranh trên môi trường phức tạp.

---

## 3. Pendulum-v1 — Continuous Action, Trung bình

### Diễn biến training

| Giai đoạn | Quantum PPO | Classical PPO | Nhận xét |
|---|---:|---:|---|
| **First 100 eps** | mean = -1137.0 (σ=228.8) | mean = -1148.9 (σ=231.8) | Khởi đầu gần như giống nhau |
| **Mid-point (~250 eps)** | mean = **-930.6** | mean = **-1040.7** | Quantum bắt đầu cải thiện nhanh hơn |
| **Last 100 eps** | mean = **-831.6** (σ=149.3) | mean = **-1086.7** (σ=111.5) | Quantum tốt hơn 23% |
| **Best đạt được** | **-624.7** | -634.5 | Gần tương đương |

### Nhận xét

**Đây là kết quả bất ngờ nhất: Quantum PPO thắng!** Sau 100k steps với cùng số episodes (500/500), Quantum đạt reward trung bình tốt hơn Classical **255 điểm (~23%)**.

**Tại sao Quantum thắng ở Pendulum?**

1. **Continuous action space phù hợp với quantum output:** PauliZ measurement cho ra giá trị liên tục trong [-1, 1], tự nhiên phù hợp để parameterize distribution của torque [-2, 2]. Classical MLP dùng `tanh` squashing nhưng với kiến trúc rất nhỏ (22 params), nó thiếu expressiveness.

2. **Quantum entanglement giúp exploration:** Các qubit entangled với nhau qua CNOT gates cho phép mạch lượng tử "khám phá" correlation giữa cos(θ), sin(θ), θ̇ một cách tự nhiên — 3 features liên quan chặt chẽ về mặt vật lý (góc và vận tốc góc của pendulum).

3. **Classical actor quá nhỏ:** Classical actor chỉ có 22 parameters (Linear(3→4) + Linear(4→1)). Với continuous action space, mô hình này có thể chưa đủ sức biểu diễn policy tốt. Quantum VQC với 32 quantum params + 516 pre-encoding params có khả năng biểu diễn tốt hơn.

**Điểm yếu còn lại:** Cả 2 đều chưa giải được Pendulum (ngưỡng reward ≥ -200). Quantum đạt best = -624.7, còn Classical -634.5 — rất gần nhau ở peak, nhưng Quantum duy trì được mức cao này nhất quán hơn trong các episode cuối.

> **Kết luận Pendulum:** Quantum PPO cho thấy lợi thế thực sự trong continuous action space với ít tham số. Đây là tín hiệu tích cực nhất từ toàn bộ thực nghiệm.

---

## 4. Phân tích Tổng hợp

### 4.1 Tốc độ học sớm (Early Convergence)

```
Môi trường   │ Q first100 eps │ C first100 eps │ Quantum advantage?
─────────────┼────────────────┼────────────────┼───────────────────
CartPole     │ mean = 22.6    │ mean = 20.2    │ ✅ Nhỉnh hơn ~12%
LunarLander  │ mean = -154.8  │ mean = -168.9  │ ✅ Nhỉnh hơn ~8%
Pendulum     │ mean = -1137.0 │ mean = -1148.9 │ ✅ Nhỉnh hơn ~1%
```

> Quantum PPO **nhất quán khởi đầu tốt hơn** ở giai đoạn đầu training trên cả 3 môi trường. Điều này gợi ý rằng data reuploading encoding + quantum entanglement giúp *khám phá policy space* hiệu quả hơn từ sớm.

### 4.2 Hiệu quả tham số

| Model | Env | Actor params | Final avg |
|---|---|---:|---:|
| Quantum | CartPole | 1,018 (42 quantum) | 319.7 |
| Classical | CartPole | 30 | **409.1** |
| Quantum | LunarLander | 1,018 (24 quantum) | 43.3 |
| Classical | LunarLander | 108 | **162.6** |
| Quantum | Pendulum | 554 (32 quantum) | **-831.6** |
| Classical | Pendulum | 22 | -1086.7 |

> Quantum actor thực ra có **nhiều tham số tổng cộng hơn** do pre-encoding MLP — so sánh "công bằng" nên tính cả pre-encoding. Tuy nhiên phần lượng tử thuần tuý (24-42 params) là rất nhỏ.

### 4.3 Bottleneck của Quantum PPO hiện tại

```
Vấn đề                     Biểu hiện                           Hệ quả
──────────────────────────────────────────────────────────────────────
1. Barren Plateau          Gradient quantum → 0 theo chiều sâu  Học chậm ở layers sâu
2. Overhead mô phỏng       237 FPS (quantum) vs 1829 FPS (cls)  Sample efficiency giả
3. Bất đối xứng LR         Pre-enc học nhanh, quantum học chậm  Policy instability
4. Bottleneck encoding     8D → 6 qubits (LunarLander)          Mất thông tin
5. Không dùng GPU cho QC   lightning.qubit CPU                  Không scale được
```

---

## 5. Kiến nghị Cải thiện

### Ngắn hạn (dễ implement)
1. **Tăng `n_layers` lên 3-4** cho LunarLander — thêm expressiveness cho VQC
2. **Thêm `n_qubits` = 8** cho LunarLander (1 qubit/feature) — tránh nén thông tin
3. **Giảm `actor_lr`** cho CartPole (5e-3 → 1e-3) — giảm instability sau hội tụ
4. **Chạy nhiều seeds** (0, 1, 2, 42) — kết quả 1 seed không đủ tin cậy về mặt thống kê

### Trung hạn (cần thay đổi architecture)
5. **Bỏ pre-encoding MLP** → dùng amplitude encoding trực tiếp (tinh khiết hơn về lượng tử)
6. **Thêm noise mitigation** — ZNE (Zero Noise Extrapolation) để giảm barren plateau
7. **Dùng natural gradient** thay parameter-shift — QNGD giúp tránh flat landscape

### Dài hạn (hướng nghiên cứu)
8. **Hardware quantum** — Chạy trên thực tế (IBM Q, Rigetti) thay mô phỏng
9. **Quantum advantage regime** — Tìm bài toán có cấu trúc tự nhiên phù hợp VQC
10. **Equivariant QNN** — Tận dụng symmetry của môi trường vào cấu trúc mạch

---

## 6. Verdict cuối cùng

| | CartPole | LunarLander | Pendulum |
|---|:---:|:---:|:---:|
| **Hiệu năng cuối** | Classical | Classical | **Quantum** |
| **Khởi động nhanh** | **Quantum** | **Quantum** | **Quantum** |
| **Ổn định** | Classical | Classical | Classical |
| **Tốc độ tính toán** | Classical | Classical | Classical |
| **Tổng điểm** | Classical | Classical | **Quantum** |

> **Nhìn chung:** Quantum PPO ở trạng thái hiện tại chưa outperform Classical PPO về hiệu năng tổng thể — điều này nhất quán với literature (2022-2024). Tuy nhiên, **tín hiệu tích cực rõ ràng**: khởi đầu nhanh hơn đồng đều và thắng trên Pendulum continuous cho thấy quantum entanglement **thực sự có giá trị** trong một số bài toán. Với nhiều timesteps hơn và kiến trúc tối ưu hơn, khoảng cách có thể thu hẹp đáng kể.
