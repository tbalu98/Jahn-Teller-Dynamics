import matplotlib.pyplot as plt


print(plt.rcParams['font.family'])

plt.xlim(0,40)
plt.ylim(0,40)


"""
plt.text(1.7, 36.75, r'$	\hat{H}_{\text{DJT}} = F\left(\hat{X} \otimes \sigma_z - \hat{Y}\otimes\sigma_x \right) + G \left[ \left( \hat{X}^2 - \hat{Y}^2 \right)\otimes \sigma_z + 2\hat{X}\hat{Y}\otimes\sigma_x \right] $', fontsize = 30)


plt.text(1.7, 33.0, r'$\hat{H}_{\text{ext}} = \mu_{\text{Bohr}}g_{L}B_{z} \hat{L}_z + \mu_{\text{Bohr}} g \left( B_x\hat{S}_x + B_y\hat{S}_y + B_z\hat{S}_z  \right)$', fontsize = 30)


plt.text(1.7, 28, r'$ \hat{H}_{\text{vib}} = \hbar\omega\left( a_x^\dagger a_x + a_y^\dagger a_y + 1 \right) $', fontsize = 30)

plt.text(1.7, 23, r'$ \hat{X} = \dfrac{ a_x^\dagger + a_x}{\sqrt{2}} $', fontsize = 30)

plt.text(1.7, 16, r'$ \hat{Y} = \dfrac{ a_y^\dagger + a_y}{\sqrt{2}} $', fontsize = 30)

plt.text(1.7, 10.0, r'$ \lambda_{\text{theory}} = p\lambda_{\text{DFT}} $', fontsize = 30)


plt.text(1.7, 8.0, r'$ \hat{H}_{\text{SOC}} = \lambda_{\text{DFT}} \hat{L}\otimes\hat{S} $', fontsize = 30)

plt.text(1.7, 4.0, r'$\hat{H} =  \hat{H}_{\text{vib}} + \hat{H}_{\text{SOC}} + \hat{H}_{\text{DJT}}  + \hat{H}_{\text{ext}}$', fontsize = 30)

"""

#plt.text(1.7, 8.0, r'$ \hat{H}_{\text{SOC}} = \lambda_{\text{DFT}} \hat{L}\otimes\hat{S} $', fontsize = 30)

#plt.text(1.8, 8.0, r'$    \lambda_{\text{theory}} = \langle\varepsilon_{3,4} |\hat{H} | \varepsilon_{3,4}\rangle - \langle\varepsilon_{1,2} |\hat{H}| \varepsilon_{1,2}\rangle$', fontsize = 30)


#plt.text(1.8, 4.0, r'$         |E_{\frac{1}{2}}\rangle= |e^{\text{DFT}}_{+}\rangle\otimes|\downarrow\rangle,  |e^{\text{DFT}}_{-}\rangle\otimes|\uparrow\rangle$', fontsize = 30)

#plt.text(1.8, 4.0, r'$ \lambda_{\text{exp}} \approx \lambda_{\text{theory}} = p\lambda_{DFT}$', fontsize = 30)

plt.text(1.8, 4.0, r'$\hat{S}_x  \hat{S}_y  \hat{S}_z $', fontsize = 30)

plt.text(1.8, 14.0, r'$\vec{a}  \vec{b}  \vec{c} $', fontsize = 30)

plt.axis("off")


plt.show()
