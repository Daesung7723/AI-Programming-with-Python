import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from matplotlib.widgets import Slider, Button

# --- 1. 초기 파라미터 및 전역 변수 설정 ---
initial_samples = 500
initial_epochs = 50
initial_lr = 0.05

X, y = [], []
scaler = StandardScaler()
history = []
current_learning_rate = initial_lr
contour_plot = None

# --- 2. 그래프 창 및 위젯 기본 틀 설정 ---
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.3)

scatter = ax.scatter([], [], c=[], cmap='viridis', edgecolors='k', alpha=0.7, zorder=3)

ax.grid(True)
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')

# 위젯(슬라이더, 버튼)들의 위치를 지정하고 생성
ax_sample_slider = plt.axes([0.15, 0.15, 0.3, 0.03])
sample_slider = Slider(ax=ax_sample_slider, label='Sample Size', valmin=20, valmax=500, valinit=initial_samples, valstep=10)

ax_epoch_slider = plt.axes([0.15, 0.08, 0.3, 0.03])
epoch_slider = Slider(ax=ax_epoch_slider, label='Epochs', valmin=1, valmax=initial_epochs, valinit=initial_epochs, valstep=1)

lr_label_ax = plt.axes([0.55, 0.15, 0.3, 0.03]); lr_label_ax.axis('off')
lr_label_ax.text(0.5, 0.5, 'Learning Rate', ha='center', va='center')

ax_minus_btn = plt.axes([0.55, 0.08, 0.08, 0.04]); minus_button = Button(ax_minus_btn, '-')
lr_text_ax = plt.axes([0.65, 0.08, 0.1, 0.04]); lr_text_ax.axis('off')
lr_text_display = lr_text_ax.text(0.5, 0.5, f'{current_learning_rate:.4f}', ha='center', va='center')
ax_plus_btn = plt.axes([0.77, 0.08, 0.08, 0.04]); plus_button = Button(ax_plus_btn, '+')


# --- 3. 핵심 기능 함수 정의 ---

def rerun_training(event=None):
    """데이터 양이나 학습률이 바뀔 때, 학습 전체를 다시 실행하는 메인 함수"""
    global X, y, scaler, history
    num_samples = int(sample_slider.val)
    
    # --- CHANGE HERE: Increased cluster_std for more overlap ---
    X, y = make_blobs(n_samples=num_samples, centers=2, n_features=2, random_state=42, cluster_std=2.8)
    # -----------------------------------------------------------
    
    scatter.set_offsets(np.c_[X[:, 0], X[:, 1]])
    scatter.set_array(y)
    
    x_margin = (X[:, 0].max() - X[:, 0].min()) * 0.1
    y_margin = (X[:, 1].max() - X[:, 1].min()) * 0.1
    ax.set_xlim(X[:, 0].min() - x_margin, X[:, 0].max() + x_margin)
    ax.set_ylim(X[:, 1].min() - y_margin, X[:, 1].max() + y_margin)
    
    X_scaled = scaler.fit_transform(X)
    
    model = SGDClassifier(loss='log_loss', warm_start=True, eta0=current_learning_rate, penalty=None, tol=None)
    history = []
    unique_classes = np.unique(y)
    for epoch in range(epoch_slider.valmax):
        model.partial_fit(X_scaled, y, classes=unique_classes)
        history.append((model.coef_.copy(), model.intercept_.copy()))

    update_epoch(epoch_slider.val)

def update_epoch(val):
    """에포크 슬라이더만 움직일 때 호출되는 업데이트 함수"""
    global contour_plot
    epoch = int(epoch_slider.val)
    if not history or epoch > len(history): return

    coef, intercept = history[epoch - 1]

    temp_model = SGDClassifier(loss='log_loss'); temp_model.classes_ = np.array([0, 1])
    temp_model.coef_, temp_model.intercept_ = coef, intercept
    
    if contour_plot:
        for coll in contour_plot.collections:
            coll.remove()
            
    x_min, x_max = ax.get_xlim(); y_min, y_max = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_scaled = scaler.transform(np.c_[xx.ravel(), yy.ravel()])
    Z = temp_model.predict(grid_scaled).reshape(xx.shape)
    contour_plot = ax.contourf(xx, yy, Z, alpha=0.2, cmap='viridis', zorder=1)

    ax.set_title(f'Samples: {len(X)} | Epoch: {epoch} | Learning Rate: {current_learning_rate:.4f}')
    fig.canvas.draw_idle()

def on_plus_clicked(event):
    global current_learning_rate
    current_learning_rate *= 1.5
    lr_text_display.set_text(f'{current_learning_rate:.4f}')
    rerun_training()

def on_minus_clicked(event):
    global current_learning_rate
    current_learning_rate /= 1.5
    lr_text_display.set_text(f'{current_learning_rate:.4f}')
    rerun_training()

# --- 4. 위젯과 함수 연결 및 실행 ---
sample_slider.on_changed(rerun_training)
epoch_slider.on_changed(update_epoch)
plus_button.on_clicked(on_plus_clicked)
minus_button.on_clicked(on_minus_clicked)

rerun_training()
plt.show()