import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.preprocessing import StandardScaler
from matplotlib.widgets import Slider, Button

# --- Initial Parameters ---
initial_samples = 500
initial_epochs = 50
initial_lr = 0.01

# --- Global variables to hold state ---
X_orig, y = [], []
scaler = StandardScaler()
history = []
current_learning_rate = initial_lr

# --- Plotting Setup ---
fig, ax = plt.subplots(figsize=(10, 8))
# Adjust layout to make more room at the bottom for all the widgets
plt.subplots_adjust(bottom=0.3)

# Create placeholder plot elements that will be updated
scatter = ax.scatter([], [], alpha=0.3, label='Original Data')
ground_truth_line, = ax.plot([], [], color='black', linestyle='--', linewidth=2, label='Optimal Line (Ground Truth)')
sgd_line, = ax.plot([], [], color='red', linewidth=3, label='SGD Regression Line')

ax.grid(True)
ax.set_xlabel('Study Hours')
ax.set_ylabel('Exam Score')
ax.legend(loc='lower right')

# --- Create Interactive Widgets in a Horizontal Layout ---

# Left side: Sliders for Sample Size and Epochs
ax_sample_slider = plt.axes([0.15, 0.15, 0.3, 0.03])
sample_slider = Slider(
    ax=ax_sample_slider, label='Sample Size', valmin=10, valmax=500,
    valinit=initial_samples, valstep=10
)

ax_epoch_slider = plt.axes([0.15, 0.08, 0.3, 0.03])
epoch_slider = Slider(
    ax=ax_epoch_slider, label='Epochs', valmin=1, valmax=initial_epochs,
    valinit=initial_epochs, valstep=1
)

# Right side: Buttons and Text for Learning Rate
# Label for the buttons
lr_label_ax = plt.axes([0.55, 0.15, 0.3, 0.03])
lr_label_ax.axis('off')
lr_label_ax.text(0.5, 0.5, 'Learning Rate', ha='center', va='center')

# The buttons and text display
ax_minus_btn = plt.axes([0.55, 0.08, 0.08, 0.04])
minus_button = Button(ax_minus_btn, '-')

lr_text_ax = plt.axes([0.65, 0.08, 0.1, 0.04])
lr_text_ax.axis('off')
lr_text_display = lr_text_ax.text(0.5, 0.5, f'{current_learning_rate:.4f}', ha='center', va='center')

ax_plus_btn = plt.axes([0.77, 0.08, 0.08, 0.04])
plus_button = Button(ax_plus_btn, '+')


# --- Update Functions (No changes in logic) ---

def rerun_training(event=None):
    """
    This master function runs when SAMPLE SIZE or LEARNING RATE changes.
    It recalculates everything from scratch.
    """
    global X_orig, y, scaler, history

    num_samples = int(sample_slider.val)
    
    # 1. Regenerate data
    X_orig = 1 + 9 * np.random.rand(num_samples, 1)
    y = 5 + 10 * X_orig + np.random.randn(num_samples, 1) * 8
    
    # 2. Update plot data and rescale axes
    scatter.set_offsets(np.c_[X_orig, y])
    ax.relim()
    ax.autoscale_view()
    
    # 3. Recalculate Ground Truth
    gt_model = LinearRegression()
    gt_model.fit(X_orig, y)
    x_line = np.array([X_orig.min(), X_orig.max()])
    y_truth_line = gt_model.predict(x_line.reshape(-1, 1))
    ground_truth_line.set_data(x_line, y_truth_line)

    # 4. Re-run SGD training with current parameters
    X_scaled = scaler.fit_transform(X_orig)
    model = SGDRegressor(warm_start=True, eta0=current_learning_rate, penalty=None, tol=None)
    history = []
    for epoch in range(epoch_slider.valmax):
        model.partial_fit(X_scaled, y.ravel())
        history.append((model.coef_.copy(), model.intercept_.copy()))

    # 5. Trigger epoch update to reflect changes
    update_epoch(epoch_slider.val)

def update_epoch(val):
    """
    This lightweight function runs when ONLY the EPOCH slider changes.
    """
    epoch = int(epoch_slider.val)
    if not history or epoch > len(history): return

    scaled_coef, scaled_intercept = history[epoch - 1]
    
    x_mean, x_std = scaler.mean_[0], scaler.scale_[0]
    original_slope = scaled_coef[0] / x_std
    original_intercept = scaled_intercept[0] - (scaled_coef[0] * x_mean / x_std)
    
    x_line_vals = np.array([X_orig.min(), X_orig.max()])
    y_line_vals = original_slope * x_line_vals + original_intercept
    
    sgd_line.set_data(x_line_vals, y_line_vals)
    ax.set_title(f'Samples: {len(X_orig)} | Epoch: {epoch} | Learning Rate: {current_learning_rate:.4f}')
    fig.canvas.draw_idle()

def on_plus_clicked(event):
    global current_learning_rate
    current_learning_rate *= 1.5 # Increase learning rate
    lr_text_display.set_text(f'{current_learning_rate:.4f}')
    rerun_training() # Rerun everything with the new LR

def on_minus_clicked(event):
    global current_learning_rate
    current_learning_rate /= 1.5 # Decrease learning rate
    lr_text_display.set_text(f'{current_learning_rate:.4f}')
    rerun_training() # Rerun everything with the new LR

# --- Connect Widgets to Functions and Display ---
sample_slider.on_changed(rerun_training)
epoch_slider.on_changed(update_epoch)
plus_button.on_clicked(on_plus_clicked)
minus_button.on_clicked(on_minus_clicked)

# Initial setup call to draw the first plot
rerun_training()

plt.show()