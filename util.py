import torch,torch.nn.functional as F, numpy as np, torchvision.transforms as T
from torch import Tensor
from torch.nn import Module, Parameter
import matplotlib.pyplot as plt

def binomial_coeff(n, k):
    """
    Optimized computation of the binomial coefficient "n choose k".
    """
    if k < 0 or k > n:
        return 0
    if k > n - k:  # Take advantage of symmetry
        k = n - k

    coeff = 1
    for i in range(1, k + 1):
        coeff *= n
        coeff //= i
        n -= 1

    return coeff

def gradient_tv(target, is_angle, compute_type):
    '''
    Get gradient of total variation.
    Note that target is 2D tensor.
    '''
    x1 = target.diff(1, dim=1, prepend = target[:,0].unsqueeze(-1))
    y1 = target.diff(1, dim=0, prepend=target[0,:].unsqueeze(0))
    if is_angle:
        x1 = torch.where(x1 > torch.pi, x1 - torch.pi, 
                         torch.where(x1 < -torch.pi, x1 + torch.pi, x1))
        y1 = torch.where(y1 > torch.pi, y1 - torch.pi,
                            torch.where(y1 < -torch.pi, y1 + torch.pi, y1))

    if compute_type =="isotropic":
        return (x1-x1.roll(-1, dims=1) +y1 - y1.roll(-1,dims=0))*2
    elif compute_type == "anisotropic":
        x1 = torch.exp(1j*x1.angle())
        x1[:,0] = 0.
        y1 = torch.exp(1j*y1.angle())
        y1[0,:] = 0.
        return (x1-x1.roll(-1,dims=1)+y1-y1.roll(-1,dims=0))

def _wirtinger(target):
    '''
    Get Wirtinger derivative.
    Note that target is 2D tensor.
    '''
    return -(target.imag-1j*target.real)/target.abs().square()

@torch.no_grad()
def gradient_tv_abs_arg(target, lambda_abs, lambda_angle,**kwargs):
    '''
    Get gradient of total variation of the absolute value and the phase of complex tensor.
    '''
    return lambda_abs * gradient_tv_amplitude(target, **kwargs)+ lambda_angle*gradient_tv_angle(target, **kwargs)

@torch.no_grad()
def gradient_tv_amplitude(target, **kwargs):
    return gradient_tv(target.abs(),**kwargs) * torch.exp(1j*target.angle().detach())

@torch.no_grad()
def gradient_tv_angle(target,**kwargs):
    '''
    Get gradient of total variation of the phase of complex tensor.
    Note that the gradient is applied to the target, calculated by target.
    '''
    return gradient_tv(target.angle(), is_angle = True, **kwargs)*_wirtinger(target)

def my_clip(target, limit = 1.):
    '''
    Clip the absolute value of complex tensor.
    '''
    return limit * (target.abs() > limit) * torch.exp(1j* target.angle()) + (target.abs() <= limit) * target

def my_pad(target, pad_type, pad_value, mode = "constant", value = 0 ):
    '''
    Apply padding to the target.
    I made this function because I don't know how to use torch.nn.functional.pad intuitively.
    '''
    if pad_type == "ratio":
        pad_size = (int(target.shape[-2] * pad_value), int(target.shape[-1]* pad_value))
    elif pad_type == "shape":
        pad_size = pad_value
    else:
        raise ValueError(f"Invalid pad_type: {pad_type}")
    padding_top = (pad_size[0]-target.shape[-2])//2
    padding_bottom = pad_size[0] - target.shape[-2] - padding_top
    padding_left = (pad_size[-1] - target.shape[-1]) //2 
    padding_right = pad_size[1] - target.shape[-1] - padding_left

    return F.pad(target,(padding_left,padding_right,padding_top,padding_bottom),mode=mode,value=value)

def my_crop(target,to):
    '''
    Crop the target, around the center.
    '''
    h,v = target.shape[-2:]
    return target[...,h//2-to[0]//2:h//2+(to[0]+1)//2,v//2-to[1]//2:v//2+(to[1]+1)//2]

# The below two functions do not need any explanation.
def FT(x): return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x),norm="ortho"))
def iFT(x): return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x),norm="ortho"))

def tv_val(target, compute_type):
    '''
    Calculate total variation of the target.
    The result has no dimension.
    '''
    if compute_type == "isotropic":
        return target.diff(1,dim=1).abs().square().sum()+target.diff(1,dim=0).abs().square().sum()
    elif compute_type == "anisotropic":
        return (target.diff(1,dim=1).abs().sum()+target.diff(1,dim=0).abs().sum())
    
def tv_val_abs_arg(target, lambda_abs, lambda_angle, **kwargs):
    return lambda_abs * tv_val(target.abs(), **kwargs) + lambda_angle * tv_val(target.angle(), **kwargs)

def tv_val_amplitude(target, **kwargs):
    return tv_val(target.abs(), **kwargs)

def tv_val_angle(target, **kwargs):
    return tv_val(target.angle(), **kwargs)

def substitute_nan(target, value):
    return torch.where(torch.isnan(target), torch.ones_like(target)*value, target)

def substitute_inf(target, value):
    return torch.where(torch.isinf(target), torch.ones_like(target)*value, target)

def substitute_nan_inf(target, value):
    return substitute_inf(substitute_nan(target, value), value)

def substitute_model_param(model, key, value):
    '''
    Substitute the parameter of the model.
    To avoid the situation that the model has no parameter with the key, raise ValueError.
    If you just model.key = value, it will not work, since the Parameter is not a simple tensor.
    '''
    for name, param in model.named_parameters():
        if name == key:
            param.data = value
            break
    else:
        raise ValueError(f"Invalid key: {key}")

def my_slice(target, to):
    '''
    Every time I use this function, I feel that I should have used torch.narrow.
    I heard that torch.narrow is faster than indexing.
    So I made a function that uses torch.narrow.
    I do not check the real performance difference by the way.
    '''
    if isinstance(target,(Tensor, np.ndarray)):
        h,v = target.shape[-2:]
    elif isinstance(target, (list, tuple)):
        h,v = target[-2:]
    return ...,slice(h//2-to[0]//2,h//2+(to[0]+1)//2),slice(v//2-to[1]//2,v//2+(to[1]+1)//2)

def my_narrow(target, to):
    '''
    This function will output the same result as my_slice.
    Note that the result will be used for the input of torch.narrow.
    Using example: torch.narrow(target, -2, *my_narrow(target.shape, to))
    Or you can use this to: target.narrow(-2, *my_narrow(target.shape, to))
    Plus, consider using this
    '''
    if isinstance(target,(Tensor, np.ndarray)):
        h,v = target.shape[-2:]
    elif isinstance(target, (list, tuple)):
        h,v = target[-2:]
    return (h//2-to[0]//2,h//2+(to[0]+1)//2),(v//2-to[1]//2,v//2+(to[1]+1)//2)

def my_interpolate(target, size, mode = "bilinear", align_corners = False):
    '''
    This function is made to interpolate the tensor.
    I made this function because I don't know how to use torch.nn.functional.interpolate intuitively. (like torch.nn.functional.pad)
    By the way, it is okay to use torch.nn.functional.interpolate, since it is more intuitive than pad.
    But I cannot get why torch.nn.functional.interpolate needs the input to be 4D tensor.
    '''
    if len(target.shape) == 2:
        input_ = target.unsqueeze(0).unsqueeze(0)
        return F.interpolate(input_, size = size, mode = mode, align_corners = align_corners).squeeze(0).squeeze(0)
    elif len(target.shape) == 3:
        input_ = target.unsqueeze(0)
        return F.interpolate(input_, size = size, mode = mode, align_corners = align_corners).squeeze(0)
    elif len(target.shape) == 4:
        return F.interpolate(target, size = size, mode = mode, align_corners = align_corners)
    else:
        raise ValueError(f"Invalid shape: {target.shape}")

def my_interpolate_complex(target, size, mode = "bilinear", align_corners = False):
    return my_interpolate(target.real, size, mode, align_corners) + 1j * my_interpolate(target.imag, size, mode, align_corners)

def span(*args,**kwargs):
    device = kwargs.get("device", "cpu")
    dtype = kwargs.get("dtype", torch.float32)
    if len(args) == 1:
        if isinstance(args[0], (list, tuple)):
            if len(args[0])==2:
                H,V = args[0]
                resol = kwargs.get("resol",1)
            else:
                raise ValueError(f"Invalid args: {args}")
    elif len(args) == 2:
        if isinstance(args[0],(list,tuple)):
            if len(args[0])==2:
                H,V = args[0]
            else:
                raise ValueError(f"Invalid args: {args}")
            resol = kwargs.get("resol",args[1])
        else:
            H,V = args
            resol = kwargs.get("resol",1)
    else:
        if isinstance(args[0],(list,tuple)):
            if len(args[0])==2:
                H,V = args[0]
            else:
                raise ValueError(f"Invalid args: {args}")
            resol = kwargs.get("resol",args[1])
        else:
            H,V = args[:2]
            resol = kwargs.get("resol",args[2])
    resol_x = kwargs.get("resol_x",resol)
    resol_y = kwargs.get("resol_y",resol)
    xbase = torch.linspace(-(V//2),(V-1)//2,V,device=device,dtype=dtype)
    ybase = torch.linspace(-(H//2),(H-1)//2,H,device=device,dtype=dtype)
    x = xbase * resol_x
    y = ybase * resol_y
    fx = xbase / resol_x / V
    fy = ybase / resol_y / H
    Fx,Fy = torch.meshgrid(fx,fy,indexing="xy")
    X,Y = torch.meshgrid(x,y,indexing="xy")
    R = torch.sqrt(X**2+Y**2)
    Fr = torch.sqrt(Fx**2+Fy**2)
    return X,Y,R,Fx,Fy,Fr

def normalized_cross_correlation(a: Tensor, b: Tensor):
    '''
    Get normalized cross_correlation of two tensors.
    Note that two tensors are 2D tensors.
    Maximum value of the output is 1.
    '''
    if a.shape != b.shape:
        raise ValueError(f"Invalid shape: {a.shape}, {b.shape}")
    a = a - a.mean(dim=(-2,-1),keepdim=True)
    b = b - b.mean(dim=(-2,-1),keepdim=True)
    return (a*b).mean(dim=(-2,-1),keepdim=True) / (a.std(dim=(-2,-1),keepdim=True) * b.std(dim=(-2,-1),keepdim=True))

def fft_normalized_cross_correlation(a: Tensor, b: Tensor):
    '''
    Get normalized cross_correlation of two tensors.
    The output is 2D tensor.
    Using fft, so it shows the registration error.
    '''
    if a.shape != b.shape:
        raise ValueError(f"Invalid shape: {a.shape}, {b.shape}")
    a = a - a.mean(dim=(-2,-1),keepdim=True)
    b = b - b.mean(dim=(-2,-1),keepdim=True)
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.fft2(a)*torch.fft.fft2(b).conj(),norm="ortho")).real

def get_shift(original, shifted, resol = 1., resol_x = None, resol_y = None):
    ncc = fft_normalized_cross_correlation(original, shifted)
    H,V = original.shape[-2:]
    resol_x = resol if resol_x is None else resol_x
    resol_y = resol if resol_y is None else resol_y
    return resol_x * (ncc.argmax(-1) - V//2), resol_y * (ncc.argmax(-2) - H//2)

def get_shift_optimizer(original, shifted, resol = 1., resol_x = None, resol_y = None, optimizer = "adam", lr = 1e-2, max_iter = 1000, **kwargs):
    if optimizer == "adam":
        optimizer = torch.optim.Adam
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD
    else:
        raise ValueError(f"Invalid optimizer: {optimizer}")
    H,V = original.shape[-2:]
    resol_x = resol if resol_x is None else resol_x
    resol_y = resol if resol_y is None else resol_y
    x = torch.zeros(1,device=original.device,dtype=original.dtype)
    y = torch.zeros(1,device=original.device,dtype=original.dtype)
    optimizer = optimizer([x,y],lr=lr)
    for i in range(max_iter):
        optimizer.zero_grad()
        shifted_ = my_interpolate(shifted, (H,V), **kwargs)
        loss = -fft_normalized_cross_correlation(original, shifted_).mean()
        loss.backward()
        optimizer.step()
    return x * resol_x, y * resol_y

def subpixel_shift(target, dist, slice_pos, resol = None, F= None, resol_x = None, resol_y = None, device = "cpu"):
    if F is None:
        H,V = target.shape[-2:]
        resol_x = resol if resol_x is None else resol_x
        resol_y = resol if resol_y is None else resol_y
        _,_,_,Fx,Fy,_ = span(H,V,resol_x=resol_x,resol_y=resol_y,device=device)
        F = torch.cat((Fx.unsqueeze(0),Fy.unsqueeze(0)),dim=0)
    else:
        if F.shape[-2:] != target.shape[-2:]:
            raise ValueError(f"Invalid shape: {F.shape}")
    F = F.to(device)
    F_input = FT(target)
    ramp = get_ramp(F,dist)
    return iFT(F_input*ramp).real[...,slice_pos[0],slice_pos[1]]

def get_ramp(F,dist):
    return torch.exp(2j*np.pi*F*dist.unsqueeze(-1).unsqueeze(-1))

Z_list = [
    "1",
    "cos_PHI*Rnorm",
    "sin_PHI*Rnorm",
    "cos_2PHI*Rsquare",
    "2*Rsquare-1",
    "sin_2PHI*Rsquare",
    "cos_3PHI*Rcube",
    "cos_PHI*(3*Rcube-2*Rnorm)",
    "sin_PHI*(3*Rcube-2*Rnorm)",
    "sin_3PHI*Rcube",
    "cos_4PHI*R4sqare",
    "cos_2PHI*(4*R4sqare-3*Rsquare)",
    "6*R4sqare-6*Rsquare+1",
    "sin_2PHI*(4*R4sqare-3*Rsquare)",
    "sin_4PHI*R4sqare",
    "cos_5PHI*R5square",
    "cos_3PHI*(5*R5square-4*Rcube)",
    "cos_PHI*(10*R5square-12*Rcube+3*Rnorm)",
    "sin_PHI*(10*R5square-12*Rcube+3*Rnorm)",
    "sin_3PHI*(5*R5square-4*Rcube)",
    "sin_5PHI*R5square",
    "cos_6PHI*R6square",
    "cos_4PHI*(6*R6square-5*R4sqare)",
    "cos_2PHI*(15*R6square-20*R4sqare+6*Rsquare)",
    "20*R6square-30*R4sqare+12*Rsquare-1",
    "sin_2PHI*(15*R6square-20*R4sqare+6*Rsquare)",
    "sin_4PHI*(6*R6square-5*R4sqare)",
    "sin_6PHI*R6square",
    "cos_7PHI*R7square",
    "cos_5PHI*(7*R7square-6*R5square)",
    "cos_3PHI*(21*R7square-30*R5square+10*Rcube)",
    "cos_PHI*(35*R7square-60*R5square+30*Rcube-4*Rnorm)",
    "sin_PHI*(35*R7square-60*R5square+30*Rcube-4*Rnorm)",
    "sin_3PHI*(21*R7square-30*R5square+10*Rcube)",
    "sin_5PHI*(7*R7square-6*R5square)",
    "sin_7PHI*R7square",
    "cos_8PHI*R8square",
    "cos_6PHI*(8*R8square-7*R6square)",
    "cos_4PHI*(28*R8square-42*R6square+15*R4sqare)",
    "cos_2PHI*(70*R8square-140*R6square+90*R4sqare-20*Rsquare)",
    "70*R8square-140*R6square+90*R4sqare-20*Rsquare+1",
    "sin_2PHI*(70*R8square-140*R6square+90*R4sqare-20*Rsquare)",
    "sin_4PHI*(28*R8square-42*R6square+15*R4sqare)",
    "sin_6PHI*(8*R8square-7*R6square)",
    "sin_8PHI*R8square",
    "cos_9PHI*R9square",
    "cos_7PHI*(9*R9square-8*R7square)",
    "cos_5PHI*(36*R9square-56*R7square+21*R5square)",
    "cos_3PHI*(84*R9square-168*R7square+105*R5square-20*Rcube)",
    "cos_PHI*(126*R9square-280*R7square+210*R5square-60*Rcube+5*Rnorm)",
    "sin_PHI*(126*R9square-280*R7square+210*R5square-60*Rcube+5*Rnorm)",
    "sin_3PHI*(84*R9square-168*R7square+105*R5square-20*Rcube)",
    "sin_5PHI*(36*R9square-56*R7square+21*R5square)",
    "sin_7PHI*(9*R9square-8*R7square)",
    "sin_9PHI*R9square",
    "cos_10PHI*R10square",
    "cos_8PHI*(10*R10square-9*R8square)",
    "cos_6PHI*(45*R10square-72*R8square+28*R6square)",
    "cos_4PHI*(120*R10square-240*R8square+180*R6square-45*R4sqare)",
    "cos_2PHI*(252*R10square-630*R8square+560*R6square-210*R4sqare+30*Rsquare)",
    "252*R10square-630*R8square+560*R6square-210*R4sqare+30*Rsquare-1",
    "sin_2PHI*(252*R10square-630*R8square+560*R6square-210*R4sqare+30*Rsquare)",
    "sin_4PHI*(120*R10square-240*R8square+180*R6square-45*R4sqare)",
    "sin_6PHI*(45*R10square-72*R8square+28*R6square)",
    "sin_8PHI*(10*R10square-9*R8square)",
    "sin_10PHI*R10square",
]
def zernike(coeff, shape=(1024,1024),portion=1,R=None,PHI=None):
    if R is None:
        X,Y,R,_,_,_ = span(shape,resol=1/(shape[0]//2))
    if PHI is None:
        PHI = (X+1j*Y).angle()
    ring_flat = (R<portion)
    Rnorm = R/portion * ring_flat
    PHI = PHI * ring_flat
    len_coeff = len(coeff) if isinstance(coeff,(list,tuple)) else coeff.shape[-1]
    cos_PHI = torch.cos(PHI)
    sin_PHI = torch.sin(PHI)
    if len_coeff >= 4:
        cos_2PHI = torch.cos(2*PHI)
        sin_2PHI = torch.sin(2*PHI)
        Rsquare = Rnorm**2
    if len_coeff >= 7:
        cos_3PHI = torch.cos(3*PHI)
        sin_3PHI = torch.sin(3*PHI)
        Rcube = Rnorm**3
    if len_coeff >= 11:
        cos_4PHI = torch.cos(4*PHI)
        sin_4PHI = torch.sin(4*PHI)
        R4sqare = Rnorm**4
    if len_coeff >= 16:
        cos_5PHI = torch.cos(5*PHI)
        sin_5PHI = torch.sin(5*PHI)
        R5square = Rnorm**5
    if len_coeff >= 22:
        cos_6PHI = torch.cos(6*PHI)
        sin_6PHI = torch.sin(6*PHI)
        R6square = Rnorm**6
    if len_coeff >= 29:
        cos_7PHI = torch.cos(7*PHI)
        sin_7PHI = torch.sin(7*PHI)
        R7square = Rnorm**7
    if len_coeff >= 37:
        cos_8PHI = torch.cos(8*PHI)
        sin_8PHI = torch.sin(8*PHI)
        R8square = Rnorm**8
    if len_coeff >= 46:
        cos_9PHI = torch.cos(9*PHI)
        sin_9PHI = torch.sin(9*PHI)
        R9square = Rnorm**9
    if len_coeff >= 56:
        cos_10PHI = torch.cos(10*PHI)
        sin_10PHI = torch.sin(10*PHI)
        R10square = Rnorm**10
    if len_coeff >= 67:
        raise ValueError(f"Too long co_eff: {len_coeff}")
    for i in range(len_coeff):
        if i == 0:
            if isinstance(coeff,(list,tuple)):
                Z = coeff[0]
            else:
                Z = coeff[...,0]
        else:
            Z += coeff[i] * eval(Z_list[i])
    return Z


def get_transfer_function(field_size, resol, wl, prop, device="cpu"):
    Fx,Fy,Fr = span(field_size[-2:],resol=resol,device=device)[3:]
    x_fov = resol * field_size[-1]
    y_fov = resol * field_size[-2]
    fx_max = x_fov/(wl*(x_fov**2+2*prop**2)**(1/2))
    fy_max = y_fov/(wl*(y_fov**2+2*prop**2)**(1/2))
    Fcut = (Fx.abs() <= fx_max)*(Fy.abs() <= fy_max)
    Fcut = Fcut * (Fr < wl**(-1))
    Fr = Fcut * Fr
    return torch.exp(2j*np.pi*prop * torch.sqrt(wl**(-2)-Fr**2))*Fcut

def angular_spectrum_method(field, resol, wl, prop, device="cpu", pad = True, pad_ratio = None, pad_size = None,
                            transfer_function = None, out_slice = None, **kwargs):
    field_size = field.shape[-2:]
    if pad:
        if pad_ratio is not None:
            field = my_pad(field, "ratio", pad_ratio, **kwargs)
        elif pad_size is not None:
            field = my_pad(field, "shape", pad_size, **kwargs)
        else:
            raise ValueError("Invalid pad")
        pad_slice = my_slice(field, field_size)
    if transfer_function is None:
        transfer_function = get_transfer_function(field.shape, resol, wl, prop, device=device)
    
    if out_slice is None:
        return iFT(FT(field)*transfer_function)[...,pad_slice[0],pad_slice[1]] if pad else iFT(FT(field)*transfer_function)
    else:
        return iFT(FT(field)*transfer_function)[...,out_slice[0],out_slice[1]]
    
class AngularSpectrumMethod(Module):
    def __init__(self,field_size, resol, wl, prop,
                 pad = True, pad_ratio = None, pad_size = None, requires_grad = False, device = "cpu", **kwargs):
        super().__init__()
        self._field_size = field_size
        self._resol = resol
        self._wl = wl
        self._prop = Parameter(torch.tensor(prop),requires_grad=requires_grad)
        self._pad = pad
        self._pad_ratio = pad_ratio
        self._pad_size = pad_size
        self = self.to(device)
    
    def forward(self, field, out_slice = None, **kwargs):
        return angular_spectrum_method(field, self._resol, self._wl, self._prop, pad = self._pad, pad_ratio = self._pad_ratio, pad_size = self._pad_size,
                                      transfer_function = self.transfer_function, out_slice = out_slice, **kwargs)
    
    def back(self, field, out_slice = None, **kwargs):
        return angular_spectrum_method(field, self._resol, self._wl, -self._prop, pad = self._pad, pad_ratio = self._pad_ratio, pad_size = self._pad_size,
                                      transfer_function = self.back_transfer_function, out_slice = out_slice, **kwargs)
    
    def update_transfer_function(self):
        if self._pad:
            if self._pad_size is not None:
                pass
            elif self._pad_ratio is not None:
                self._pad_size = (int(self._field_size[-2]*self._pad_ratio),int(self._field_size[-1]*self._pad_ratio))
            else:
                pad_radius = minimum_pad_size(self._field_size, self._resol, self._wl, self._prop, device=self._prop.device)
                self._pad_size = (2*pad_radius+self._field_size[-2],2*pad_radius+self._field_size[-1])
            self.register_buffer("transfer_function", get_transfer_function(self._pad_size, self._resol, self._wl, self._prop, device=self._prop.device))
            self.register_buffer("back_transfer_function", get_transfer_function(self._pad_size, self._resol, self._wl, -self._prop, device=self._prop.device))
        else:
            self.register_buffer("transfer_function", get_transfer_function(self._field_size, self._resol, self._wl, self._prop, device=self._prop.device))
            self.register_buffer("back_transfer_function", get_transfer_function(self._field_size, self._resol, self._wl, -self._prop, device=self._prop.device))
    
    def update_new_transfer_function(self, new_transfer_function):
        """
        Update the transfer function.
        """
        self.transfer_function = new_transfer_function

    @property
    def prop(self):
        return self._prop.item()
    @prop.setter
    def prop(self, value):
        self._prop.data = torch.tensor(value,device=self._prop.device,dtype=self._prop.dtype)
        self.update_transfer_function()
    @property
    def field_size(self):
        return self._field_size
    @field_size.setter
    def field_size(self, value):
        self._field_size = value
        self.update_transfer_function()
    @property
    def resol(self):
        return self._resol
    @resol.setter
    def resol(self, value):
        self._resol = value
        self.update_transfer_function()
    @property
    def wl(self):
        return self._wl
    @wl.setter
    def wl(self, value):
        self._wl = value
        self.update_transfer_function()
    @property
    def pad(self):
        return self._pad
    @pad.setter
    def pad(self, value):
        self._pad = value
        self.update_transfer_function()
    @property
    def pad_ratio(self):
        return self._pad_ratio
    @pad_ratio.setter
    def pad_ratio(self, value):
        self._pad_ratio = value
        self.update_transfer_function()
    @property
    def pad_size(self):
        return self._pad_size
    @pad_size.setter
    def pad_size(self, value):
        self._pad_size = value
        self.update_transfer_function()
    
def minimum_pad_size(field_size, resol, wl, prop, threshold = 0.95, transfer_function = None,device = "cpu"):
    if transfer_function is None:
        transfer_function = get_transfer_function(field_size, resol, wl, prop, device=device)
    
    fft_tf = FT(transfer_function)
    intensity = fft_tf.abs().pow(2)
    R = span(field_size,resol=1,device=device)[2]
    radial_intensity = torch.stack([intensity[R<i].sum() for i in range(1,int(R.max())+1)])
    threshold_index = torch.searchsorted(radial_intensity, radial_intensity[-1]*threshold)
    print(f"Minimum radius: {threshold_index.item()} given threshold: {threshold}")
    return threshold_index.item()

class SubPixelShift(Module):
    def __init__(self, input_sz = None, resol = None, crop_to = None, slice_pos = None, resol_x = None, resol_y = None, device = "cpu"):
        super().__init__()
        if input_sz is None:
            raise ValueError("input_sz must be given")
        if resol is None:
            raise ValueError("resol must be given")
        self.input_size = input_sz
        self.resol = resol
        if (crop_to is None) and (slice_pos is None):
            raise ValueError("crop_to or slice_pos must be given")
        if crop_to is not None:
            self.slice_pos = my_slice(input_sz, crop_to)
        if slice_pos is not None:
            self.slice_pos = slice_pos
        self.resol_x = resol if resol_x is None else resol_x
        self.resol_y = resol if resol_y is None else resol_y
        self.device = device
        self.to(device)
        Fx,Fy = span(input_sz,resol=resol,device=device)[3:5]
        self.register_buffer("F", torch.stack((Fx,Fy),dim=0))
    
    def forward(self, target, dist):
        return subpixel_shift(target, dist, self.slice_pos, resol = self.resol, F = self.F, resol_x = self.resol_x, resol_y = self.resol_y, device = self.device)

def get_resol_xy_of_sample_for_conical_correction(img, angle, wl, prop, pix, x_cut_type = "default"):
    H,V = img.shape[-2:]
    init_max_freq = torch.sin(torch.arctan(torch.tensor(pix*H/2/prop)))/wl
    init_resol = 1/(init_max_freq*2)
    Fx,Fy = span((H//2,V//2),resol = init_resol,device=img.device)[3:5]
    sin_a = torch.sin(torch.tensor(angle))
    freq_x = wl * Fx + sin_a
    freq_y = wl * Fy
    fr = torch.sqrt(freq_x**2+freq_y**2)
    if sin_a>0:
        test_fn = torch.max
    else:
        test_fn = torch.min
    if x_cut_type == "default":
        freq_x_for_resol = test_fn(Fx*(fr<1))
    elif x_cut_type == "tight":
        freq_x_for_resol = test_fn((Fx*(fr<1))[0,:])
    elif x_cut_type == "loose":
        freq_x_for_resol = test_fn((Fx*(fr<1))) * 1.5
    else:
        raise ValueError(f"Invalid x_cut_type: {x_cut_type}")
    return torch.abs(1/(freq_x_for_resol*2)), init_resol

def angle_validation(angle, angle_type):
    if angle>0:
        if angle_type == "degree":
            _angle = (90-angle)*torch.pi/180
        elif angle_type == "radian":
            _angle = torch.pi/2-angle
        else:
            raise ValueError(f"Invalid angle_type: {angle_type}")
    else:
        if angle_type == "degree":
            _angle = -(90+angle)*torch.pi/180
        elif angle_type == "radian":
            _angle = -(torch.pi/2+angle)
        else:
            raise ValueError(f"Invalid angle_type: {angle_type}")
    return _angle

def get_img_halfsize_ready_for_conical_correction(target, center,max_sz = None):
    H,V = target.shape[-2:]
    cy,cx = center
    left = min(cy,H-cy-1)
    top = min(cx,V-cx-1)
    sz = min(left,top)*2 + 1
    if max_sz is not None:
        sz = min(sz,max_sz)
    half_size = sz//2
    subtensor = target[...,cy-half_size:cy+half_size+1,cx-half_size:cx+half_size+1]
    return subtensor, half_size

def get_XY_of_camera_from_Fs_of_sample_coordinates_for_conical_correction(Fx,Fy,angle,wl,prop):
    sin_a = torch.sin(torch.tensor(angle))
    cos_a = torch.cos(torch.tensor(angle))

    freq_x = wl * Fx + sin_a
    freq_y = wl * Fy

    valid_region = (freq_x**2+freq_y**2<1)

    freq_x[~valid_region] = torch.nan
    freq_y[~valid_region] = torch.nan
    
    freq_z = torch.sqrt(1-freq_x**2-freq_y**2)

    freq = torch.stack((freq_x,freq_y,freq_z),dim=0)

    zero_diffraction_order = torch.tensor([sin_a,0,cos_a],device=freq.device).unsqueeze(-1).unsqueeze(-1)
    projection_vector = prop/torch.sum(zero_diffraction_order*freq,dim=0)

    x = torch.tensor([cos_a,0,-sin_a],device=freq.device).unsqueeze(-1).unsqueeze(-1)
    X = torch.sum((projection_vector*freq-prop*zero_diffraction_order)*x,dim=0)
    Y = (projection_vector*freq-prop*zero_diffraction_order)[1,...]

    return X,Y

def get_grid_for_conical_correction(img, angle, wl, half_size, prop, pix, x_cut_type = "tight", angle_type = "degree"):
    _angle = angle_validation(angle, angle_type)
    resol_x, resol_y = get_resol_xy_of_sample_for_conical_correction(img, _angle, wl, prop, pix, x_cut_type = x_cut_type)
    Fx,Fy = span(img.shape[-2:],resol_x = resol_x, resol_y = resol_y, device=img.device)[3:5]
    X,Y = get_XY_of_camera_from_Fs_of_sample_coordinates_for_conical_correction(Fx,Fy,_angle,wl,prop)
    return torch.concat(((X/(pix*half_size)).unsqueeze(-1),(Y/(pix*half_size)).unsqueeze(-1)),dim=-1).unsqueeze(0)

def get_intensity_corr_for_conical_correction(half_size, pix, prop):
    X,Y = span((half_size*2,half_size*2),resol=pix)[0:2]
    intensity_correction = 1+(X**2+Y**2)/(prop**2)
    return intensity_correction

def conical_correction(img_list, angle, wl, prop, pix, cy, cx, x_cut_type = "tight", angle_type = "degree"):
    _img_list, half_size = get_img_halfsize_ready_for_conical_correction(img_list, (cy,cx))
    grid = get_grid_for_conical_correction(_img_list, angle, wl, half_size, prop, pix, x_cut_type = x_cut_type, angle_type = angle_type)
    intensity_correction = get_intensity_corr_for_conical_correction(half_size, pix, prop)
    if len(_img_list.shape)==3:
        return F.grid_sample((_img_list*intensity_correction).unsqueeze(0), grid, mode="bilinear", align_corners=True).squeeze(0)
    elif len(_img_list.shape)==2:
        return F.grid_sample((_img_list*intensity_correction).unsqueeze(0).unsqueeze(0), grid, mode="bilinear", align_corners=True).squeeze(0).squeeze(0)
    elif len(_img_list.shape)==4:
        return F.grid_sample((_img_list*intensity_correction), grid, mode="bilinear", align_corners=True)
    else:
        raise ValueError(f"Invalid shape: {_img_list.shape}")
    
def my_max_index_of_img(tensor):
    if len(tensor.shape) == 2:
        return (tensor.abs()==tensor.abs().max()).nonzero()[0]
    elif len(tensor.shape) == 3:
        for i in range(tensor.shape[0]):
            if i == 0:
                max_index = my_max_index_of_img(tensor[i,...]).unsqueeze(0)
            else:
                max_index = torch.cat((max_index,my_max_index_of_img(tensor[i,...]).unsqueeze(0)),dim=0)

        return max_index
    else:
        raise ValueError(f"Invalid shape: {tensor.shape}")

class tilt_correction(Module):
    def __init__(self, img_list, wl, prop, pix):
        super().__init__()
        self.img_list = img_list
        self.wl = wl
        self.prop = prop
        self.pix = pix
        self.center_idx = self.get_center_idx()
    
    def get_center_idx(self):
        max_index = my_max_index_of_img(self.img_list)
        return torch.mean(max_index,dim=0)
    
    def test_with_show(self, angle, idx, x_cut_type = "tight", angle_type = "degree"):
        result = conical_correction(self.img_list[idx], angle, self.wl, self.prop, self.pix, self.center_idx[0], self.center_idx[1], x_cut_type = x_cut_type, angle_type = angle_type)
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(torch.log10(result),cmap="gray")
        plt.axis("off")
        plt.subplot(1,2,2)
        plt.imshow(torch.log10(result)[result.shape[0]//2-100,result.shape[0]//2+100,
                                       result.shape[1]//2-100:result.shape[1]//2+100],cmap="gray")
        plt.axis("off")
        plt.suptitle(f"Corrected image, idx: {idx}, angle: {angle}")
        plt.show()
        plt.pause(0.01)
        return result

    def forward(self, angle, x_cut_type = "tight", angle_type = "degree"):
        return conical_correction(self.img_list, angle, self.wl, self.prop, self.pix, self.center_idx[0], self.center_idx[1], x_cut_type = x_cut_type, angle_type = angle_type)


def get_freq_list_from_back_data(img_list,resol, wl, scale_factor = 4, device = "cpu"):
    H,V = img_list.shape[-2:]
    X,Y,R,Fx,Fy,Fr = span((H*scale_factor,V*scale_factor),resol=resol,device=device)
    Fimg = FT(my_pad(img_list[0],"shape",(H*scale_factor,V*scale_factor))) * (R>resol*200) # To ensure the zero frequency is not included
    #find the first maximum frequency at img_list[0], which is a result from normal illumination.
    base_freq_idx = my_max_index_of_img(Fimg)
    print(f"Index of the maximum in the normal incident condition: Y: {base_freq_idx[0].item()}, X: {base_freq_idx[1].item()}")
    ramp = torch.exp((2j*torch.pi*(X*Fx[base_freq_idx[0],base_freq_idx[1]]+Y*Fy[base_freq_idx[0],base_freq_idx[1]])))
    freq_list = torch.empty(img_list.shape[0],2)
    print("Finding the maximum frequency for each image...")
    for i, img in enumerate(img_list):
        Fimg = FT(my_pad(img,"shape",(H*scale_factor,V*scale_factor))/ramp) * (Fr<1/wl) # To ensure physical frequency range
        freq_idx = my_max_index_of_img(Fimg)
        freq_list[i,0] = Fx[freq_idx[0],freq_idx[1]].item()
        freq_list[i,1] = Fy[freq_idx[0],freq_idx[1]].item()
        print(f"Percetage: {i/img_list.shape[0]*100:.2f}%, Y: {freq_idx[0].item()}, X: {freq_idx[1].item()}")
    return freq_list

def median_filter(img,kernel_size = 3):
    if kernel_size%2 == 0:
        raise ValueError(f"Invalid kernel_size: {kernel_size} should be odd")
    pad = kernel_size//2
    img_padded = F.pad(img,(pad,pad,pad,pad),"reflect")

    patches = img_padded.unfold(2,kernel_size,1).unfold(3,kernel_size,1)
    patches = patches.permute(0,2,3,1,4,5).reshape(-1,kernel_size*kernel_size)

    values,_ = patches.sort(dim=1)
    median = values[:, kernel_size*kernel_size//2].reshape_as(img)
    return median
