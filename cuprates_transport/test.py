#%
from copy import deepcopy

class MyBandStructure():
    def __init__(self, a, b, c, band_params, energy_scale=200):
        self.a = a
        self.b = b
        self.c = c
        self.energy_scale = energy_scale
        self.band_params = band_params ## reference problem!
        self.band_params = deepcopy(band_params)

    def __setitem__(self, key, value):
        self.erase_fermi_surface()
        self.band_params[key] = value

    def __getitem__(self, key):
        return self.params_dict[key]

    def erase_fermi_surface(self):
        print("yeah!")

    def set_band_param(self, key, val):
        self[key]=val

params = {
    "a": 3.75,
    "b": 3.75,
    "c": 13.2,
    "energy_scale": 190,
    "band_params": {'tp': 0.3}
}
bs1 = MyBandStructure(**params)
bs2 = MyBandStructure(**params)
bs1['tp'] = 0.17
print(bs1.band_params)
print(bs2.band_params)
print(params)
bs1.a = 0.17
print(bs1.a)
print(bs2.a)
print(params)

bs3 = MyBandStructure(a=1, b=1, c=2, band_params={"p1": 1, "p2":-0.136, "p3":89})
bs4 = MyBandStructure(a=1, b=1, c=2, band_params={"p1": 1, "p2":-0.136, "p3":89})
bs3['tp'] = 0.17
print(bs3.band_params)
print(bs4.band_params)

## we want


