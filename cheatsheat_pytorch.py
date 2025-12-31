"""PyTorch Core Knowledge Cheatsheet - Enhanced Edition
Matplotlib-style cheatsheet for Colab - Optimized for Quick Reference
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Style settings
plt.style.use('default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.unicode_minus'] = False

# ============== COLOR PALETTE ==============
# Semantic color scheme for better visual hierarchy
COLORS = {
    # Code blocks (blue theme - most common)
    'code': '#3498DB',
    'code_bg': '#EBF5FB',
    
    # Output/results (green theme)
    'output': '#27AE60',
    'output_bg': '#E8F8F5',
    
    # Warnings/Errors (red theme)
    'warning': '#E74C3C',
    'warning_bg': '#FDEDEC',
    
    # Accent/Purple (important notes)
    'accent': '#9B59B6',
    'accent_bg': '#F5EEF8',
    
    # Titles
    'title': '#2C3E50',
    
    # Navigation/Header
    'nav_bg': '#34495E',
    'nav_text': '#ECF0F1',
    
    # Visualization colors
    'tensor': '#E67E22',
    'dim': '#16A085',
    'element': '#1ABC9C',
}

# Helper function for code blocks
def add_code_block(ax, code, x=0.5, y=9.5, color=COLORS['code'], bg_color=COLORS['code_bg']):
    ax.text(x, y, code, fontsize=8.5, family='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=bg_color, 
                     edgecolor=color, linewidth=2))

# Helper function for output blocks
def add_output_block(ax, output, x=0.5, y=9.5):
    ax.text(x, y, output, fontsize=8.5, family='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['output_bg'],
                     edgecolor=COLORS['output'], linewidth=2))

# Helper function for warning blocks
def add_warning_block(ax, warning, x=0.5, y=9.5):
    ax.text(x, y, warning, fontsize=8.5, family='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['warning_bg'],
                     edgecolor=COLORS['warning'], linewidth=2))

# Helper function for note blocks
def add_note_block(ax, note, x=0.5, y=9.5):
    ax.text(x, y, note, fontsize=8.5, family='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['accent_bg'],
                     edgecolor=COLORS['accent'], linewidth=2))

# Helper function to draw tensor visualization
def draw_tensor_visual(ax, shape, pos=(7.5, 7), color=COLORS['tensor']):
    """Draw a simple tensor shape visualization"""
    if len(shape) == 1:
        ax.add_patch(plt.Circle(pos, 0.4, fill=False, edgecolor=color, linewidth=2))
        ax.text(pos[0], pos[1], str(shape[0]), ha='center', va='center', fontsize=9, color=color)
    elif len(shape) == 2:
        w, h = 0.8, 0.6
        rect = plt.Rectangle((pos[0]-w/2, pos[1]-h/2), w, h, 
                              fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(pos[0], pos[1], f"{shape[0]}x{shape[1]}", ha='center', va='center', 
                fontsize=9, color=color, fontweight='bold')
    else:
        # Represent 3D+ as stacked boxes
        for i in range(min(shape[0], 3)):
            offset = i * 0.25
            rect = plt.Rectangle((pos[0]-0.5+offset, pos[1]-0.3-offset), 0.6, 0.5,
                                  fill=False, edgecolor=color, linewidth=1.5, alpha=0.7)
            ax.add_patch(rect)
        ax.text(pos[0], pos[1], f"{shape}\n...", ha='center', va='center', 
                fontsize=8, color=color)

# ============== NAVIGATION HEADER ==============
def add_nav_header(fig, title, page_num, total_pages, color=COLORS['nav_bg']):
    """Add navigation header to figure"""
    fig.text(0.5, 0.97, f"PyTorch Cheatsheet | {title} | {page_num}/{total_pages}", 
             ha='center', va='top', fontsize=11, fontweight='bold',
             color=COLORS['nav_text'],
             bbox=dict(boxstyle='round,pad=0.5', facecolor=color, edgecolor='none'))

# ============== Figure 1: Tensor Creation ==============
fig1 = plt.figure(figsize=(16, 11))
fig1.suptitle('', fontsize=1)  # Hidden, using nav header instead
add_nav_header(fig1, 'Tensor Creation & Basics', 1, 6)

# 1.1 From Python Data
ax1 = fig1.add_subplot(2, 3, 1)
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('From Python Data', fontsize=11, fontweight='bold', color=COLORS['title'], pad=10)

code = """# 1D Tensor (Vector)
x = torch.tensor([1, 2, 3])
# -> tensor([1, 2, 3])

# 2D Tensor (Matrix)
x = torch.tensor([[1, 2],
                  [3, 4]])
# -> tensor([[1, 2],
#           [3, 4]])

# Float tensor
x = torch.tensor([1.5, 2.5, 3.5])

# Specify dtype
x = torch.tensor([1, 2], dtype=torch.float32)"""
add_code_block(ax1, code)

# Add tensor visualization
draw_tensor_visual(ax1, [3], (7.5, 5.5))
draw_tensor_visual(ax1, [2, 2], (7.5, 2.5))

# 1.2 Zeros, Ones, Eye
ax2 = fig1.add_subplot(2, 3, 2)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('Zeros / Ones / Eye', fontsize=11, fontweight='bold', color=COLORS['title'], pad=10)

code2 = """# Zeros tensor (3x4)
z = torch.zeros(3, 4)
# -> [[0., 0., 0., 0.],
#    [0., 0., 0., 0.],
#    [0., 0., 0., 0.]]

# Ones tensor
o = torch.ones(2, 5)

# Full with value
f = torch.full((2, 3), 7.0)

# Identity matrix (3x3)
i = torch.eye(3)
# -> [[1., 0., 0.],
#    [0., 1., 0.],
#    [0., 0., 1.]]"""
add_code_block(ax2, code2, color=COLORS['accent'], bg_color=COLORS['accent_bg'])

# 1.3 Random Tensors
ax3 = fig1.add_subplot(2, 3, 3)
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.axis('off')
ax3.set_title('Random Tensors', fontsize=11, fontweight='bold', color=COLORS['title'], pad=10)

code3 = """# Uniform [0, 1) - U[0,1)
r = torch.rand(3, 3)

# Normal (mean=0, std=1) - N(0,1)
n = torch.randn(3, 3)

# Random int [0, n)
ri = torch.randint(0, 10, (3, 3))

# Like another tensor
like_zeros = torch.zeros_like(x)
like_ones = torch.ones_like(x)

# Random permutation
p = torch.randperm(10)  # [9, 2, 5, ...]"""
add_code_block(ax3, code3, color='#27AE60', bg_color='#E8F8F5')

# 1.4 From NumPy
ax4 = fig1.add_subplot(2, 3, 4)
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')
ax4.set_title('NumPy <-> Tensor', fontsize=11, fontweight='bold', color=COLORS['title'], pad=10)

code4 = """import numpy as np

# NumPy -> Tensor (SHARES MEMORY!)
np_arr = np.array([1, 2, 3])
x = torch.from_numpy(np_arr)

x[0] = 100
print(np_arr[0])  # ! 100 (changed!)

# Explicit copy (no share)
x = torch.tensor(np_arr)
x = torch.as_tensor(np_arr)  # also copies

# Tensor -> NumPy
np_arr = x.numpy()  # also shares memory

# GPU tensor -> NumPy
np_arr = x.cpu().numpy()"""
add_code_block(ax4, code4, color='#F39C12', bg_color='#FEF9E7')

# Draw arrow for shared memory
ax4.annotate('', xy=(8, 4), xytext=(5, 4),
            arrowprops=dict(arrowstyle='<->', color=COLORS['warning'], lw=2))
ax4.text(6.5, 4.5, 'Shared Memory', fontsize=8, ha='center', color=COLORS['warning'])

# 1.5 Range & Linspace
ax5 = fig1.add_subplot(2, 3, 5)
ax5.set_xlim(0, 10)
ax5.set_ylim(0, 10)
ax5.axis('off')
ax5.set_title('Range & Linspace', fontsize=11, fontweight='bold', color=COLORS['title'], pad=10)

code5 = """# [start, end), step
x = torch.arange(0, 10, 2)
# -> [0, 2, 4, 6, 8]

# [start, end], n points (inclusive)
x = torch.linspace(0, 10, 5)
# -> [ 0.0000,  2.5000,  5.0000,
#     7.5000, 10.0000]

# Logspace (10^start to 10^end)
x = torch.logspace(0, 2, 5)
# -> [  1.,   3.16,  10.,  31.6, 100.]"""
add_code_block(ax5, code5, color='#16A085', bg_color='#E8F8F5')

# 1.6 Device & dtype
ax6 = fig1.add_subplot(2, 3, 6)
ax6.set_xlim(0, 10)
ax6.set_ylim(0, 10)
ax6.axis('off')
ax6.set_title('Device & dtype', fontsize=11, fontweight='bold', color=COLORS['title'], pad=10)

code6 = """# Common dtypes
x = torch.float32   # default
x = torch.float64   # double
x = torch.int32
x = torch.int64     # long
x = torch.bool

# Device options
device = torch.device('cuda')
device = torch.device('cpu')
device = torch.device('mps')  # Apple Silicon

# Create on device
x = torch.zeros(3, 3, device=device)

# Move tensor
x = x.to('cuda')     # shorthand
x = x.cpu()
x = x.float()        # cast dtype

# Check
x.device             # device(type='cuda', index=0)
x.dtype              # torch.float32"""
add_code_block(ax6, code6, color='#8E44AD', bg_color='#F5EEF8')

plt.tight_layout(rect=[0, 0.08, 1, 0.93])
fig1.savefig('pytorch_cheatsheet_1_creation.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

# ============== Figure 2: Tensor Operations ==============
fig2 = plt.figure(figsize=(16, 11))
add_nav_header(fig2, 'Tensor Operations & Math', 2, 6)

# 2.1 Basic Arithmetic
ax1 = fig2.add_subplot(2, 3, 1)
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('Basic Arithmetic', fontsize=11, fontweight='bold', color=COLORS['title'], pad=10)

code = """x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

# Addition
z = x + y              # -> [5, 7, 9]
z = torch.add(x, y)

# Subtraction
z = x - y              # -> [-3, -3, -3]

# Multiplication (element-wise)
z = x * y              # -> [4, 10, 18]
z = torch.mul(x, y)

# Division
z = x / y              # -> [0.25, 0.4, 0.5]
z = torch.div(x, y)

# ! In-place operations (save memory!)
x.add_(y)    # x = x + y (modified)"""
add_code_block(ax1, code)

# 2.2 Matrix Operations
ax2 = fig2.add_subplot(2, 3, 2)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('Matrix Operations', fontsize=11, fontweight='bold', color=COLORS['title'], pad=10)

code2 = """A = torch.rand(3, 4)
B = torch.rand(4, 5)

# Matrix multiplication (3x5)
C = torch.mm(A, B)
C = A @ B              # <- preferred syntax

# Element-wise multiply (3x4)
C = A * B

# Vector dot product
v1 = torch.tensor([1, 2, 3])
v2 = torch.tensor([4, 5, 6])
dot = torch.dot(v1, v2)    # -> 32

# Outer product
outer = torch.outer(v1, v2)  # (3,3)

# Cross product
cross = torch.cross(v1, v2)"""
add_code_block(ax2, code2, color=COLORS['accent'], bg_color=COLORS['accent_bg'])

# 2.3 Reshaping
ax3 = fig2.add_subplot(2, 3, 3)
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.axis('off')
ax3.set_title('Reshaping & View', fontsize=11, fontweight='bold', color=COLORS['title'], pad=10)

code3 = """x = torch.arange(12)         # [0,1,2,...,11]

# ! view (shares memory, fast)
y = x.view(3, 4)            # 3x4 matrix
y = x.view(-1, 6)           # infer dim -> 2x6

# reshape (may copy if non-contiguous)
y = x.reshape(2, 6)

# Flatten
y = x.flatten()             # -> [12]
y = x.reshape(-1)

# Squeeze / Unsqueeze
x = x.unsqueeze(0)          # add dim -> [1, 12]
x = x.squeeze()             # remove dim-1 -> [12]
x = x.unsqueeze(1)          # -> [12, 1]

# ! Important: contiguous()
x = x.view(3, 4)
if not x.is_contiguous():
    x = x.contiguous()"""
add_code_block(ax3, code3, color='#27AE60', bg_color='#E8F8F5')

# Add visual for reshape
ax3.text(7, 6.5, "view/reshape does not change data", fontsize=8, ha='center', 
         bbox=dict(boxstyle='round', facecolor='#FFF', edgecolor='#999'))
ax3.annotate('', xy=(8, 5), xytext=(5, 5),
            arrowprops=dict(arrowstyle='->', color='#999', lw=1.5, ls='--'))

# 2.4 Indexing & Slicing
ax4 = fig2.add_subplot(2, 3, 4)
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')
ax4.set_title('Indexing & Slicing', fontsize=11, fontweight='bold', color=COLORS['title'], pad=10)

code4 = """x = torch.tensor([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]])

# Basic indexing
x[0]            # -> [1, 2, 3]  (row 0)
x[:, 0]         # -> [1, 4, 7]  (col 0)
x[1, 2]         # -> 6          (element)

# Slicing
x[0:2]          # first 2 rows
x[1:, 1:]       # bottom-right 2x2

# Boolean indexing (mask)
mask = x > 5
x[mask]         # -> [6, 7, 8, 9]

# torch.where (ternary operator)
y = torch.where(x > 5, x, 0)

# gather & scatter
torch.gather(x, 1, torch.tensor([[0, 2]]))
# -> [[1, 3]]"""
add_code_block(ax4, code4, color='#F39C12', bg_color='#FEF9E7')

# 2.5 Reduction Operations
ax5 = fig2.add_subplot(2, 3, 5)
ax5.set_xlim(0, 10)
ax5.set_ylim(0, 10)
ax5.axis('off')
ax5.set_title('Reduction Operations', fontsize=11, fontweight='bold', color=COLORS['title'], pad=10)

code5 = """x = torch.tensor([[1, 2, 3],
                         [4, 5, 6]])

# Global reduction
x.sum()         # -> 21
x.mean()        # -> 3.5
x.prod()        # -> 720

# Reduction along dimension
x.sum(dim=0)    # -> [5, 7, 9]
x.mean(dim=1)   # -> [2., 5.]

# Max/Min with indices
x.max(dim=0)    # -> (values, indices)
x.min(dim=1)    # -> (values, indices)

# Argmax/Argmin
torch.argmax(x)     # -> 5 (index)
torch.argmin(x)     # -> 0 (index)

# Keep dimension
x.sum(dim=0, keepdim=True)  # -> [[5, 7, 9]]"""
add_code_block(ax5, code5, color='#16A085', bg_color='#E8F8F5')

# 2.6 Comparison Table - View vs Reshape
ax6 = fig2.add_subplot(2, 3, 6)
ax6.set_xlim(0, 10)
ax6.set_ylim(0, 10)
ax6.axis('off')
ax6.set_title('view vs reshape vs resize', fontsize=11, fontweight='bold', color=COLORS['title'], pad=10)

table_code = """+------------------+---------+-----------+
|     Method       | Memory  | Flexible  |
+------------------+---------+-----------+
| view()          | Shared  | Non-contig|
| reshape()       | May copy| Supported |
| resize_()       | May copy| Supported |
| contiguous()    | Copies  | Contig    |
+------------------+---------+-----------+

# Non-contiguous tensor cannot use view!
x = torch.randn(3, 4).t()  # transposed
y = x.view(12)  # ! Error!

# Solutions
y = x.reshape(12)  # Auto copy
y = x.contiguous().view(12)  # Manual"""
add_code_block(ax6, table_code, color=COLORS['warning'], bg_color=COLORS['warning_bg'])

plt.tight_layout(rect=[0, 0.08, 1, 0.93])
fig2.savefig('pytorch_cheatsheet_2_operations.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

# ============== Figure 3: Autograd ==============
fig3 = plt.figure(figsize=(16, 11))
add_nav_header(fig3, 'Autograd & Gradients', 3, 6)

# 3.1 Basic Gradients
ax1 = fig3.add_subplot(2, 3, 1)
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('Basic Gradients', fontsize=11, fontweight='bold', color=COLORS['title'], pad=10)

code = """x = torch.tensor([2.0], requires_grad=True)

# Forward: y = x**2 + 3x + 1
y = x**2 + 3*x + 1
# tensor([11.], grad_fn=<AddBackward0>)

# Backward pass
y.backward()

# dy/dx = 2x + 3
#     = 2(2) + 3 = 7
print(x.grad)  # -> tensor([7.])"""
add_code_block(ax1, code)

# 3.2 Weights & Biases
ax2 = fig3.add_subplot(2, 3, 2)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('Weights & Biases', fontsize=11, fontweight='bold', color=COLORS['title'], pad=10)

code2 = """w = torch.tensor([1.0], requires_grad=True)
b = torch.tensor([0.5], requires_grad=True)
x = torch.tensor([2.0])

# y = wx + b = 1*2 + 0.5 = 2.5
y = w * x + b
y.backward()

# dy/dw = x = 2.0
print(w.grad)  # -> tensor([2.])

# dy/db = 1.0
print(b.grad)  # -> tensor([1.])

# Check gradients
print(w.grad is not None)  # True"""
add_code_block(ax2, code2, color=COLORS['accent'], bg_color=COLORS['accent_bg'])

# 3.3 Zero Gradients (IMPORTANT!)
ax3 = fig3.add_subplot(2, 3, 3)
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.axis('off')
ax3.set_title('Zero Gradients (IMPORTANT!)', fontsize=11, fontweight='bold', 
              color=COLORS['warning'], pad=10)

warning = """# ! Wrong: Gradients accumulate!
for epoch in epochs:
    loss.backward()    # Gradients accumulate
    optimizer.step()
    # Forgot to zero!!

# Correct
optimizer.zero_grad()       # Method 1: Standard
w.grad = None               # Method 2: Faster
w.grad.zero_()              # Method 3: In-place

# Must call after every batch!"""
add_warning_block(ax3, warning)

# 3.4 No Grad Context
ax4 = fig3.add_subplot(2, 3, 4)
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')
ax4.set_title('No Grad Context', fontsize=11, fontweight='bold', color=COLORS['title'], pad=10)

code4 = """# Inference/eval without gradients

# Method 1: Context manager
with torch.no_grad():
    result = model(x)
    # result.requires_grad == False

# Method 2: Global switch
torch.set_grad_enabled(False)
pred = model(x)
torch.set_grad_enabled(True)  # Remember to re-enable!

# Method 3: detach (separate)
y = x.detach()
# Returns new tensor (no gradient)

# Saves memory + speeds up inference!"""
add_code_block(ax4, code4, color='#27AE60', bg_color='#E8F8F5')

# 3.5 Gradient Clipping
ax5 = fig3.add_subplot(2, 3, 5)
ax5.set_xlim(0, 10)
ax5.set_ylim(0, 10)
ax5.axis('off')
ax5.set_title('Gradient Clipping', fontsize=11, fontweight='bold', color=COLORS['title'], pad=10)

code5 = """# Prevent gradient explosion (RNN/LSTM common)

# 1. Clip by value
torch.nn.utils.clip_grad_value_(
    model.parameters(), 
    clip_value=1.0
)

# 2. Clip by norm (recommended!)
# All gradients L2 norm <= max_norm
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,
    norm_type=2  # L2 norm
)

# Usage order
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()"""
add_code_block(ax5, code5, color='#F39C12', bg_color='#FEF9E7')

# 3.6 Manual Grad with autograd
ax6 = fig3.add_subplot(2, 3, 6)
ax6.set_xlim(0, 10)
ax6.set_ylim(0, 10)
ax6.axis('off')
ax6.set_title('Manual Grad (autograd)', fontsize=11, fontweight='bold', color=COLORS['title'], pad=10)

code6 = """import torch.autograd as autograd

x = torch.tensor([3.0], requires_grad=True)
y = x**2

# Manual gradient computation
grad = autograd.grad(y, x)
# -> (tensor([6.]),)

# Multiple outputs + grad_outputs
y = x**2 + 2*x
grad = autograd.grad(
    y, x, 
    grad_outputs=torch.ones_like(x)
)
# -> (tensor([8.]),)

# create_graph=True for higher-order derivatives
grad = autograd.grad(y, x, create_graph=True)

# retain_graph=True to keep computation graph"""
add_code_block(ax6, code6, color='#8E44AD', bg_color='#F5EEF8')

plt.tight_layout(rect=[0, 0.08, 1, 0.93])
fig3.savefig('pytorch_cheatsheet_3_autograd.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

# ============== Figure 4: Neural Network ==============
fig4 = plt.figure(figsize=(16, 11))
add_nav_header(fig4, 'Neural Network (nn.Module)', 4, 6)

# 4.1 Basic Model Structure
ax1 = fig4.add_subplot(2, 3, 1)
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('Basic Model Structure', fontsize=11, fontweight='bold', color=COLORS['title'], pad=10)

code = """import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()  # ! Required!
        self.layer1 = nn.Linear(784, 256)
        self.layer2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.layer2(x)
        return x

# Create & use
model = MyModel()
print(model)  # View structure"""
add_code_block(ax1, code)

# 4.2 Common Layers
ax2 = fig4.add_subplot(2, 3, 2)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('Common Layers', fontsize=11, fontweight='bold', color=COLORS['title'], pad=10)

code2 = """# Dense / Linear
nn.Linear(in_features, out_features)

# Convolutional (2D)
nn.Conv2d(in_ch, out_ch, kernel_size=3, 
          stride=1, padding=1)
nn.MaxPool2d(kernel_size=2)
nn.AvgPool2d(kernel_size=2)
nn.AdaptiveAvgPool2d((1, 1))

# Recurrent
nn.LSTM(input_size, hidden_size, num_layers)
nn.GRU(input_size, hidden_size)

# Normalization
nn.BatchNorm2d(num_features)
nn.LayerNorm(normalized_shape)
nn.GroupNorm(num_groups, num_channels)

# Dropout
nn.Dropout(p=0.5)
nn.Dropout2d(p=0.5)"""
add_code_block(ax2, code2, color=COLORS['accent'], bg_color=COLORS['accent_bg'])

# 4.3 Activation Functions
ax3 = fig4.add_subplot(2, 3, 3)
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.axis('off')
ax3.set_title('Activation Functions', fontsize=11, fontweight='bold', color=COLORS['title'], pad=10)

code3 = """# In nn.Module (stateful)
self.act = nn.ReLU()
self.act = nn.Sigmoid()
self.act = nn.Tanh()
self.act = nn.GELU()      # Popular
self.act = nn.SiLU()      # Swish
self.act = nn.Softmax(dim=1)

# In forward (functional)
x = F.relu(x)
x = F.leaky_relu(x, negative_slope=0.01)
x = F.gelu(x)             # Recommended
x = F.silu(x)             # SiLU / Swish
x = F.softmax(x, dim=1)

# Dropout in training mode
x = F.dropout(x, p=0.5, training=model.training)"""
add_code_block(ax3, code3, color='#27AE60', bg_color='#E8F8F5')

# 4.4 Parameters & Weights
ax4 = fig4.add_subplot(2, 3, 4)
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')
ax4.set_title('Parameters & Weights', fontsize=11, fontweight='bold', color=COLORS['title'], pad=10)

code4 = """# Iterate all parameters
for param in model.parameters():
    print(param.shape)

# Named parameters
for name, param in model.named_parameters():
    print(name, param.shape)

# Access specific layer
model.layer1.weight      # Layer1 weights
model.layer1.bias        # Layer1 bias

# Direct data access
w = model.layer1.weight.data
b = model.layer1.bias.data

# Parameter count
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() 
                if p.requires_grad)"""
add_code_block(ax4, code4, color='#F39C12', bg_color='#FEF9E7')

# 4.5 Device Management
ax5 = fig4.add_subplot(2, 3, 5)
ax5.set_xlim(0, 10)
ax5.set_ylim(0, 10)
ax5.axis('off')
ax5.set_title('Device Management', fontsize=11, fontweight='bold', color=COLORS['title'], pad=10)

code5 = """# Check GPU
device = torch.device('cuda' 
    if torch.cuda.is_available() else 'cpu')
print(device)  # cuda or cpu

# Move model to device
model.to(device)

# Move data to device
inputs = inputs.to(device)
labels = labels.to(device)

# Shorthand
model = model.cuda()
data = data.cuda()

# Check tensor device
x.device  # device(type='cuda', index=0)

# DataParallel (single machine, multi-GPU)
model = nn.DataParallel(model)

# DistributedDataParallel (multi-machine, multi-GPU)
model = nn.DistributedDataParallel(model)"""
add_code_block(ax5, code5, color='#16A085', bg_color='#E8F8F5')

# 4.6 Save & Load
ax6 = fig4.add_subplot(2, 3, 6)
ax6.set_xlim(0, 10)
ax6.set_ylim(0, 10)
ax6.axis('off')
ax6.set_title('Save & Load Model', fontsize=11, fontweight='bold', color=COLORS['title'], pad=10)

code6 = """# Recommended: Save state dict only
torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth'))

# Not recommended: Save entire model
torch.save(model, 'entire_model.pth')
model = torch.load('entire_model.pth')

# Save checkpoint (full training state)
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss.item(),
}
torch.save(checkpoint, 'checkpoint.pth')

# Load checkpoint
ckpt = torch.load('checkpoint.pth')
model.load_state_dict(ckpt['model_state_dict'])
optimizer.load_state_dict(ckpt['optimizer_state_dict'])"""
add_code_block(ax6, code6, color='#8E44AD', bg_color='#F5EEF8')

plt.tight_layout(rect=[0, 0.08, 1, 0.93])
fig4.savefig('pytorch_cheatsheet_4_nn.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

# ============== Figure 5: Data Loading ==============
fig5 = plt.figure(figsize=(16, 11))
add_nav_header(fig5, 'Data Loading & Datasets', 5, 6)

# 5.1 Custom Dataset
ax1 = fig5.add_subplot(2, 3, 1)
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('Custom Dataset', fontsize=11, fontweight='bold', color=COLORS['title'], pad=10)

code = """from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data      # Features
        self.labels = labels  # Labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

# Usage
train_data = CustomDataset(X_train, y_train)
test_data = CustomDataset(X_test, y_test)"""
add_code_block(ax1, code)

# 5.2 TensorDataset
ax2 = fig5.add_subplot(2, 3, 2)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('TensorDataset', fontsize=11, fontweight='bold', color=COLORS['title'], pad=10)

code2 = """from torch.utils.data import TensorDataset

# Simple tensor wrapper
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Access
x, y = train_dataset[0]
print(x.shape, y.shape)

# Multiple tensors
dataset = TensorDataset(x1, x2, y)

# Convert to DataLoader
loader = DataLoader(dataset, batch_size=32)"""
add_code_block(ax2, code2, color=COLORS['accent'], bg_color=COLORS['accent_bg'])

# 5.3 DataLoader
ax3 = fig5.add_subplot(2, 3, 3)
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.axis('off')
ax3.set_title('DataLoader', fontsize=11, fontweight='bold', color=COLORS['title'], pad=10)

code3 = """from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,           # Random shuffle
    num_workers=4,          # Parallel loading
    pin_memory=True,        # GPU transfer speedup
    drop_last=False,        # Drop incomplete batch
    collate_fn=None,        # Custom batch processing
    prefetch_factor=2       # Prefetch
)

# Iterate
for batch_x, batch_y in train_loader:
    print(batch_x.shape)   # [32, features]
    print(batch_y.shape)   # [32]
    break

# Colab recommends num_workers=0"""
add_code_block(ax3, code3, color='#27AE60', bg_color='#E8F8F5')

# 5.4 Transforms (Images)
ax4 = fig5.add_subplot(2, 3, 4)
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')
ax4.set_title('Transforms (Images)', fontsize=11, fontweight='bold', color=COLORS['title'], pad=10)

code4 = """from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),           # [0,1]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet
        std=[0.229, 0.224, 0.225]
    ),
])

# Common augmentations
transforms.RandomHorizontalFlip(p=0.5)
transforms.RandomRotation(degrees=10)
transforms.ColorJitter(brightness=0.2)
transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
transforms.RandomErasing(p=0.1)  # Random erasing

# Use ImageFolder
from torchvision.datasets import ImageFolder
dataset = ImageFolder('data/', transform=transform)"""
add_code_block(ax4, code4, color='#F39C12', bg_color='#FEF9E7')

# 5.5 Samplers
ax5 = fig5.add_subplot(2, 3, 5)
ax5.set_xlim(0, 10)
ax5.set_ylim(0, 10)
ax5.axis('off')
ax5.set_title('Custom Samplers', fontsize=11, fontweight='bold', color=COLORS['title'], pad=10)

code5 = """from torch.utils.data import WeightedRandomSampler

# 1. WeightedRandomSampler (imbalanced data)
weights = [0.9, 0.1, 0.1, 0.8]  # Weight per sample
sampler = WeightedRandomSampler(
    weights, 
    num_samples=100, 
    replacement=True
)
loader = DataLoader(dataset, sampler=sampler)

# 2. SubsetRandomSampler (random split)
from torch.utils.data import SubsetRandomSampler
train_idx, val_idx = ...  # Split indices
train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

# 3. BatchSampler (wrap other sampler)
from torch.utils.data import BatchSampler
sampler = BatchSampler(
    RandomSampler(dataset),
    batch_size=32,
    drop_last=False
)"""
add_code_block(ax5, code5, color='#16A085', bg_color='#E8F8F5')

# 5.6 Custom Collate & Padding
ax6 = fig5.add_subplot(2, 3, 6)
ax6.set_xlim(0, 10)
ax6.set_ylim(0, 10)
ax6.axis('off')
ax6.set_title('Custom Collate (Padding)', fontsize=11, fontweight='bold', color=COLORS['title'], pad=10)

code6 = """# Variable length sequence padding
def pad_collate_fn(batch):
    data, labels = zip(*batch)
    lengths = [len(x) for x in data]
    
    # Pad sequences
    padded = torch.nn.utils.rnn.pad_sequence(
        data, 
        batch_first=True,
        padding_value=0
    )
    
    return padded, torch.tensor(labels), torch.tensor(lengths)

loader = DataLoader(
    dataset, 
    collate_fn=pad_collate_fn,
    batch_size=32
)

# Custom batch processing
def my_collate_fn(batch):
    data = [item[0] for item in batch]
    label = [item[1] for item in batch]
    
    data = torch.stack(data, dim=0)
    label = torch.tensor(label)
    
    return data, label"""
add_code_block(ax6, code6, color='#8E44AD', bg_color='#F5EEF8')

plt.tight_layout(rect=[0, 0.08, 1, 0.93])
fig5.savefig('pytorch_cheatsheet_5_data.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

# ============== Figure 6: Training Loop ==============
fig6 = plt.figure(figsize=(16, 11))
add_nav_header(fig6, 'Training Loop & Optimization', 6, 6)

# 6.1 Loss Functions
ax1 = fig6.add_subplot(2, 3, 1)
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('Loss Functions', fontsize=11, fontweight='bold', color=COLORS['title'], pad=10)

code = """import torch.nn as nn

# Classification
criterion = nn.CrossEntropyLoss()      # Multi-class
criterion = nn.BCEWithLogitsLoss()     # Binary (with sigmoid)
criterion = nn.BCELoss()               # Binary (already sigmoid)
criterion = nn.NLLLoss()               # Log probabilities

# Regression
criterion = nn.MSELoss()               # L2 (MSE)
criterion = nn.L1Loss()                # L1 (MAE)
criterion = nn.SmoothL1Loss()          # Huber

# Usage
loss = criterion(outputs, targets)
print(loss.item())  # Get scalar value

# CrossEntropyLoss expects logits (unnormalized)"""
add_code_block(ax1, code)

# 6.2 Optimizers
ax2 = fig6.add_subplot(2, 3, 2)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('Optimizers', fontsize=11, fontweight='bold', color=COLORS['title'], pad=10)

code2 = """import torch.optim as optim

# SGD (with momentum)
optimizer = optim.SGD(
    model.parameters(), 
    lr=0.01, 
    momentum=0.9,
    weight_decay=1e-4  # L2 regularization
)

# Adam (most commonly used)
optimizer = optim.Adam(
    model.parameters(), 
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=0.01
)

# AdamW (decoupled weight decay)
optimizer = optim.AdamW(
    model.parameters(), 
    lr=0.001, 
    weight_decay=0.01
)

# RMSprop
optimizer = optim.RMSprop(model.parameters(), lr=0.01)

# Adafactor (recommended for large models)
optimizer = optim.Adafactor(model.parameters())"""
add_code_block(ax2, code2, color=COLORS['accent'], bg_color=COLORS['accent_bg'])

# 6.3 Learning Rate Schedulers
ax3 = fig6.add_subplot(2, 3, 3)
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.axis('off')
ax3.set_title('LR Schedulers', fontsize=11, fontweight='bold', color=COLORS['title'], pad=10)

code3 = """# StepLR (step decay)
scheduler = optim.lr_scheduler.StepLR(
    optimizer, 
    step_size=30, 
    gamma=0.1
)

# ExponentialLR (exponential decay)
scheduler = optim.lr_scheduler.ExponentialLR(
    optimizer, 
    gamma=0.95
)

# CosineAnnealing (cosine annealing)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=100,
    eta_min=1e-6
)

# ReduceLROnPlateau (monitor metric)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',
    factor=0.1,
    patience=5
)

# OneCycleLR (super-convergence)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.1,
    epochs=10,
    steps_per_epoch=len(train_loader)
)

# Update timing: scheduler.step() after optimizer.step()"""
add_code_block(ax3, code3, color='#27AE60', bg_color='#E8F8F5')

# 6.4 Complete Training Loop
ax4 = fig6.add_subplot(2, 3, 4)
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')
ax4.set_title('Complete Training Loop', fontsize=11, fontweight='bold', color=COLORS['title'], pad=10)

code4 = """num_epochs = 10
best_acc = 0.0

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    
    for batch_x, batch_y in train_loader:
        # Key steps
        optimizer.zero_grad()           # 1. Zero gradients
        outputs = model(batch_x)        # 2. Forward pass
        loss = criterion(outputs, batch_y)  # 3. Compute loss
        loss.backward()                 # 4. Backward pass
        optimizer.step()                # 5. Update params
        scheduler.step()                # 6. Update LR
        
        train_loss += loss.item()
    
    # Validation
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():               # Disable gradients
        for val_x, val_y in val_loader:
            outputs = model(val_x)
            _, predicted = torch.max(outputs, 1)
            total += val_y.size(0)
            correct += (predicted == val_y).sum().item()
    
    acc = 100 * correct / total
    print(f'Epoch [{epoch+1}], Loss: {train_loss:.4f}, Acc: {acc:.2f}%')
    
    # Save best model
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'best_model.pth')"""
add_code_block(ax4, code4, color='#F39C12', bg_color='#FEF9E7')

# 6.5 Inference
ax5 = fig6.add_subplot(2, 3, 5)
ax5.set_xlim(0, 10)
ax5.set_ylim(0, 10)
ax5.axis('off')
ax5.set_title('Inference / Prediction', fontsize=11, fontweight='bold', color=COLORS['title'], pad=10)

code5 = """# Load best model
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Single sample inference
sample = torch.randn(1, 784)  # [1, features]

with torch.no_grad():
    output = model(sample)
    probs = torch.softmax(output, dim=1)
    pred = torch.argmax(probs, dim=1)
print(f'Prediction: {pred.item()}')

# Batch inference
with torch.no_grad():
    outputs = model(test_data)
    predictions = torch.argmax(outputs, dim=1)

# Get Top-K probabilities
probabilities = torch.softmax(outputs, dim=1)
top_probs, top_classes = probabilities.topk(k=5)
print(f'Top 5 classes: {top_classes[0]}')

# Save inference model
torch.save(model.state_dict(), 'inference_model.pth')"""
add_code_block(ax5, code5, color='#16A085', bg_color='#E8F8F5')

# 6.6 Common Pitfalls
ax6 = fig6.add_subplot(2, 3, 6)
ax6.set_xlim(0, 10)
ax6.set_ylim(0, 10)
ax6.axis('off')
ax6.set_title('Common Pitfalls', fontsize=11, fontweight='bold', 
              color=COLORS['warning'], pad=10)

pitfalls = """# Common Mistakes

# 1. Forgot .zero_grad()
loss.backward()
optimizer.step()  # Gradients not zeroed!

# 2. Inference without no_grad
model.eval()
output = model(input)  # But recommend with torch.no_grad()

# 3. Wrong input to CrossEntropyLoss
output = model(x)  # logits
loss = criterion(output, target)  # Correct

# 4. Model not .to(device)
model = MyModel()
output = model(data)  # data on GPU, model on CPU!

# 5. Forgot model.train()/eval()
model.eval()  # Call for inference
model.train()  # Call for training

# 6. Gradient accumulation wrong batch_size
loss = loss / accumulation_steps  # Maintain scale"""
add_warning_block(ax6, pitfalls)

plt.tight_layout(rect=[0, 0.08, 1, 0.93])
fig6.savefig('pytorch_cheatsheet_6_training.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

# ============== Figure 7: Advanced Topics (NEW!) ==============
fig7 = plt.figure(figsize=(16, 11))
add_nav_header(fig7, 'Advanced: torch.compile & Mixed Precision', 7, 7)

# 7.1 torch.compile (PyTorch 2.0+)
ax1 = fig7.add_subplot(2, 3, 1)
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('torch.compile (PyTorch 2.0+)', fontsize=11, 
              fontweight='bold', color=COLORS['title'], pad=10)

code = """# PyTorch 2.0+ features
# Compile model for significant speedup

# Basic usage
from torch import compile

model = MyModel()
model = compile(model)  # One-line speedup!

# Compile options
model = compile(
    model,
    mode='default',       # default/maxmin/reduce-overhead
    fullgraph=True,       # Ensure no graph breaks
    dynamic=False         # Static graph (more stable)
)

# Check if compiled
print(model._compiled_call_impl)  # True if compiled

# First run compiles, subsequent runs are faster"""
add_code_block(ax1, code, color='#E74C3C', bg_color='#FDEDEC')

# 7.2 Mixed Precision (AMP)
ax2 = fig7.add_subplot(2, 3, 2)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('Mixed Precision (AMP)', fontsize=11, 
              fontweight='bold', color=COLORS['title'], pad=10)

code2 = """# Mixed precision training (save memory + speedup)

# Method 1: torch.cuda.amp (PyTorch 1.6+)
scaler = torch.cuda.amp.GradScaler()

for batch in train_loader:
    optimizer.zero_grad()
    
    with torch.cuda.amp.autocast():  # Auto FP16
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
    
    scaler.scale(loss).backward()    # Scale loss
    
    scaler.step(optimizer)           # Optimizer step
    scaler.update()                  # Update scale factor

# Method 2: torch.autocast (cast only, no scale)
with torch.autocast(device_type='cuda', dtype=torch.float16):
    outputs = model(inputs)
    loss = criterion(outputs, labels)

# Save ~50% memory, speedup 20-50%"""
add_code_block(ax2, code2, color='#9B59B6', bg_color='#F5EEF8')

# 7.3 Distributed Training
ax3 = fig7.add_subplot(2, 3, 3)
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.axis('off')
ax3.set_title('Distributed Training', fontsize=11, 
              fontweight='bold', color=COLORS['title'], pad=10)

code3 = """# Multi-GPU training

# 1. Single machine, multi-GPU - DataParallel (simple)
model = nn.DataParallel(model)

# 2. Single machine, multi-GPU - DDP (recommended)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

# Initialize
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

# Create model
model = model.cuda(local_rank)
model = DDP(model, device_ids=[local_rank])

# Data sampler
sampler = DistributedSampler(dataset)
loader = DataLoader(dataset, sampler=sampler)

# Training loop
for batch in loader:
    loss = model(batch).sum()
    loss.backward()
    optimizer.step()

# Launch command
# torchrun --nproc_per_node=4 train.py"""
add_code_block(ax3, code3, color='#27AE60', bg_color='#E8F8F5')

# 7.4 Gradient Accumulation
ax4 = fig7.add_subplot(2, 3, 4)
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')
ax4.set_title('Gradient Accumulation', fontsize=11, 
              fontweight='bold', color=COLORS['title'], pad=10)

code4 = """# Not enough memory? Use gradient accumulation!

accumulation_steps = 4  # Accumulate 4 batches

for epoch in range(num_epochs):
    model.train()
    
    for i, (batch_x, batch_y) in enumerate(train_loader):
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Scale loss (optional)
        loss = loss / accumulation_steps
        
        loss.backward()
        
        # Update every accumulation_steps steps
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

# effective_batch_size = batch_size * accumulation_steps"""
add_code_block(ax4, code4, color='#F39C12', bg_color='#FEF9E7')

# 7.5 Efficient Finetuning
ax5 = fig7.add_subplot(2, 3, 5)
ax5.set_xlim(0, 10)
ax5.set_ylim(0, 10)
ax5.axis('off')
ax5.set_title('Efficient Finetuning', fontsize=11, 
              fontweight='bold', color=COLORS['title'], pad=10)

code5 = """# Efficient finetuning techniques

# 1. Frozen Backbone
for param in model.parameters():
    param.requires_grad = False
# Only train classifier
for param in model.classifier.parameters():
    param.requires_grad = True

# 2. Layer-wise Learning Rate
optimizer = optim.Adam([
    {'params': model.backbone.parameters(), 'lr': 1e-4},
    {'params': model.classifier.parameters(), 'lr': 1e-3},
])

# 3. LoRA (Low-Rank Adaptation)
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.05,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()"""
add_code_block(ax5, code5, color='#16A085', bg_color='#E8F8F5')

# 7.6 Quick Reference Card
ax6 = fig7.add_subplot(2, 3, 6)
ax6.set_xlim(0, 10)
ax6.set_ylim(0, 10)
ax6.axis('off')
ax6.set_title('Quick Commands', fontsize=11, fontweight='bold', color=COLORS['title'], pad=10)

quick_ref = """# Common Commands Quick Reference

# Version check
print(torch.__version__)

# GPU check
torch.cuda.is_available()
torch.cuda.device_count()
torch.cuda.get_device_name(0)

# Memory management
torch.cuda.empty_cache()
torch.cuda.memory_allocated()
torch.cuda.max_memory_allocated()

# Model info
model.eval()              # Inference mode
model.train()             # Training mode
model.parameters()        # All parameters
model.named_children()    # Submodules

# Tensor info
x.shape, x.size(), x.dtype, x.device
x.numel(), x.nelement()   # Element count
x.is_cuda, x.requires_grad
x.is_contiguous()

# Random seed (reproducibility)
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True"""
add_code_block(ax6, quick_ref, color='#8E44AD', bg_color='#F5EEF8')

plt.tight_layout(rect=[0, 0.08, 1, 0.93])
fig7.savefig('pytorch_cheatsheet_7_advanced.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

# ============== Summary ==============
print("\n" + "="*70)
print("PyTorch Cheatsheet Enhanced Edition Generated!")
print("="*70)
print("\nFiles generated:")
print("   1. pytorch_cheatsheet_1_creation.png      - Tensor Creation & Basics")
print("   2. pytorch_cheatsheet_2_operations.png    - Tensor Operations & Math")
print("   3. pytorch_cheatsheet_3_autograd.png      - Autograd & Gradients")
print("   4. pytorch_cheatsheet_4_nn.png            - Neural Network Module")
print("   5. pytorch_cheatsheet_5_data.png          - Data Loading")
print("   6. pytorch_cheatsheet_6_training.png      - Training Loop")
print("   7. pytorch_cheatsheet_7_advanced.png      - Advanced Topics")
print("\nNew Features:")
print("   - Unified semantic color scheme")
print("   - 7 pages complete content (Advanced page added)")
print("   - Tensor visualization aids")
print("   - Common mistakes guide")
print("   - Comparison tables (view vs reshape)")
print("   - Distributed training (DDP)")
print("   - torch.compile speedup guide")
print("   - Mixed precision training (AMP)")
print("   - Efficient finetuning (LoRA)")
print("   - Gradient accumulation tricks")
print("="*70)