import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.preprocessing import StandardScaler
from matplotlib.widgets import Slider, Button

class RegressionVisualizer:
    def __init__(self, ax, initial_samples=500, initial_epochs=50, initial_lr=0.01):
        self.ax = ax
        self.num_samples = initial_samples
        self.max_epochs = initial_epochs
        self.learning_rate = initial_lr

        self.X_orig, self.y = [], []
        self.scaler = StandardScaler()
        self.history = []

        # --- Plotting Setup ---
        self.scatter = self.ax.scatter([], [], alpha=0.3, label='Original Data')
        self.ground_truth_line, = self.ax.plot([], [], color='black', linestyle='--', linewidth=2, label='Optimal Line (Ground Truth)')
        self.sgd_line, = self.ax.plot([], [], color='red', linewidth=3, label='SGD Regression Line')
        self.ax.grid(True)
        self.ax.set_xlabel('Study Hours')
        self.ax.set_ylabel('Exam Score')
        self.ax.legend(loc='lower right')

    def _unscale_coeffs(self, scaled_coef, scaled_intercept):
        """Helper to convert scaled model coefficients back to original data space."""
        x_mean, x_std = self.scaler.mean_[0], self.scaler.scale_[0]
        # Handle cases where std is zero to avoid division by zero
        if x_std == 0:
            return 0, scaled_intercept[0]
            
        original_slope = scaled_coef[0] / x_std
        original_intercept = scaled_intercept[0] - (scaled_coef[0] * x_mean / x_std)
        return original_slope, original_intercept

    def rerun_training(self, event=None):
        """Recalculates everything when sample size or learning rate changes."""
        self.num_samples = int(sample_slider.val)
        
        # 1. Regenerate data
        self.X_orig = 1 + 9 * np.random.rand(self.num_samples, 1)
        self.y = 5 + 10 * self.X_orig + np.random.randn(self.num_samples, 1) * 8
        
        # 2. Update plot data and rescale axes
        self.scatter.set_offsets(np.c_[self.X_orig, self.y])
        self.ax.relim()
        self.ax.autoscale_view()
        
        # 3. Recalculate Ground Truth
        gt_model = LinearRegression()
        gt_model.fit(self.X_orig, self.y)
        x_line = np.array([self.X_orig.min(), self.X_orig.max()])
        y_truth_line = gt_model.predict(x_line.reshape(-1, 1))
        self.ground_truth_line.set_data(x_line, y_truth_line)

        # 4. Re-run SGD training
        X_scaled = self.scaler.fit_transform(self.X_orig)
        model = SGDRegressor(warm_start=True, eta0=self.learning_rate, penalty=None, max_iter=1, tol=None)
        self.history = []
        for _ in range(self.max_epochs):
            model.partial_fit(X_scaled, self.y.ravel())
            self.history.append((model.coef_.copy(), model.intercept_.copy()))

        # 5. Trigger epoch update to reflect changes
        self.update_epoch(epoch_slider.val)

    def update_epoch(self, val):
        """Updates the plot for a given epoch without retraining."""
        epoch = int(val)
        if not self.history or epoch > len(self.history): return

        scaled_coef, scaled_intercept = self.history[epoch - 1]
        
        original_slope, original_intercept = self._unscale_coeffs(scaled_coef, scaled_intercept)
        
        x_line_vals = np.array([self.X_orig.min(), self.X_orig.max()])
        y_line_vals = original_slope * x_line_vals + original_intercept
        
        self.sgd_line.set_data(x_line_vals, y_line_vals)
        self.ax.set_title(f'Samples: {len(self.X_orig)} | Epoch: {epoch} | Learning Rate: {self.learning_rate:.4f}')
        fig.canvas.draw_idle()

    def on_plus_clicked(self, event):
        self.learning_rate *= 1.5
        lr_text_display.set_text(f'{self.learning_rate:.4f}')
        self.rerun_training()

    def on_minus_clicked(self, event):
        self.learning_rate /= 1.5
        lr_text_display.set_text(f'{self.learning_rate:.4f}')
        self.rerun_training()

# --- Main Execution ---
if __name__ == '__main__':
    # --- Initial Parameters ---
    initial_samples = 500
    initial_epochs = 50
    initial_lr = 0.01

    # --- Plotting Setup ---
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.3)

    # --- Create Visualizer Instance ---
    visualizer = RegressionVisualizer(ax, initial_samples, initial_epochs, initial_lr)

    # --- Create Interactive Widgets ---
    ax_sample_slider = plt.axes([0.15, 0.15, 0.3, 0.03])
    sample_slider = Slider(
        ax=ax_sample_slider, label='Sample Size', valmin=10, valmax=1000,
        valinit=initial_samples, valstep=10
    )

    ax_epoch_slider = plt.axes([0.15, 0.08, 0.3, 0.03])
    epoch_slider = Slider(
        ax=ax_epoch_slider, label='Epochs', valmin=1, valmax=initial_epochs,
        valinit=initial_epochs, valstep=1
    )

    lr_label_ax = plt.axes([0.55, 0.15, 0.3, 0.03])
    lr_label_ax.axis('off')
    lr_label_ax.text(0.5, 0.5, 'Learning Rate', ha='center', va='center')

    ax_minus_btn = plt.axes([0.55, 0.08, 0.08, 0.04])
    minus_button = Button(ax_minus_btn, '-')

    lr_text_ax = plt.axes([0.65, 0.08, 0.1, 0.04])
    lr_text_ax.axis('off')
    lr_text_display = lr_text_ax.text(0.5, 0.5, f'{initial_lr:.4f}', ha='center', va='center')

    ax_plus_btn = plt.axes([0.77, 0.08, 0.08, 0.04])
    plus_button = Button(ax_plus_btn, '+')

    # --- Connect Widgets to Visualizer Methods ---
    sample_slider.on_changed(visualizer.rerun_training)
    epoch_slider.on_changed(visualizer.update_epoch)
    plus_button.on_clicked(visualizer.on_plus_clicked)
    minus_button.on_clicked(visualizer.on_minus_clicked)

    # --- Initial setup call ---
    visualizer.rerun_training()

    plt.show()