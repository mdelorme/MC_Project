# Base imports
print('Importing packages ...')
import numpy as np
import sys
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy.coordinates import SkyCoord
from astropy import units as u
from ebv import get_SFD_dust

## Leavitt Law, coefficients come from Storm et al. 2011
ll_a = {'V': -2.73, 'I': -2.91, 'J': -3.19, 'K': -3.30}
ll_b = {'V': -3.97, 'I': -4.75, 'J': -5.20, 'K': -5.65}

# PL relationship, gives absolute magnitude from a band and a period
LL = lambda P, band : ll_a[band]*(np.log10(P)-1.0) + ll_b[band]

## Extinctions coming from the CMD-Padova website : http://stev.oapd.inaf.it/cgi-bin/cmd
## The values for 2Mass JHK come from Cohen et al 2003
## The values for OGLE-IV I/V indicated in Vanhollebeke et al 2009
Al_Av = {'J': 0.29434, 'H': 0.18128, 'K': 0.11838, 'V': 1.02093, 'I': 0.57082}

Rv = 3.23 # Coming from Sandage 2004

## A class to encapsulate everything
class Dataset(object):
    def __init__(self, filename, reddening_file=None, compute_extinction=False):
        ''' Constructor '''
        
        print('Creating a new dataset, extracted from {} :'.format(filename))
        print('  . Reading input file')
        self.filename = filename
        self.raw_data = ascii.read(filename)

        # Extracting sky coordinates in degrees
        self.ra  = np.asarray(self.raw_data['ra'])
        self.dec = np.asarray(self.raw_data['dec'])

        # Number of points
        self.N = self.ra.shape[0]

        # Photometry
        self.V_band = np.asarray(self.raw_data['V'], dtype=np.float32)
        self.I_band = np.asarray(self.raw_data['V'], dtype=np.float32)

        # Period
        self.period = np.asarray(self.raw_data['P'], dtype=np.float32)

        # Computing extinction, if deactivated we skip this whole thing
        if compute_extinction:
            # This operation is a bit long so this can be skipped if we provide a reddening_file
            # as input of the constructor.
            if reddening_file == None:
                print('  . Computing reddening')
                
                # Creating a coordinate object to get galactocentric coordinates from Astropy
                c = SkyCoord(ra=self.ra*u.degree, dec=self.dec*u.degree)
                l, b = c.galactic.l, c.galactic.b

                # Using the ebv package to get reddening from a dust map
                self.ebv = get_SFD_dust(l, b, dustmap='data/SFD_dust_4096_sgp.fits')
            else:
                # We already have reddening calculated somewhere
                ebv_file = ascii.read(reddening_file)
                self.ebv = np.asarray(ebv_file['E_B_V_SandF'])
        else:
            self.ebv = 0.0

        # We make this computation here, instead of having to declare all these values to 0
        # to simplify the rest of the code. If no extinction is to be computed, then self.ebv=0
        self.AV = Al_Av['V'] * Rv * self.ebv
        self.AI = Al_Av['I'] * Rv * self.ebv

        ## Distances
        print('  . Computing distances')
        # Magnitudes from PL
        self.M_V = LL(self.period, 'V')
        self.M_I = LL(self.period, 'I')
        # Distance modulus
        self.mu_V = self.V_band - self.M_V - self.AV
        self.mu_I = self.I_band - self.M_I - self.AI
        # And finally distances in kpc
        self.d_V = 10**(1.0 + self.mu_V / 5.0) * 1.0e-3
        self.d_I = 10**(1.0 + self.mu_I / 5.0) * 1.0e-3

        # Registering all numpy arrays so we can mask the data
        # This is a bit tricky, it is to make sure that if we want to remove some stars from the dataset
        # these stars are easily removed from ALL the arrays we have just created
        # Once all these arrays are registered, we can use the mask function to remove stars from specific criteria
        print('  . Registering arrays :')
        self.registered_arrays = []
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if type(attr) == np.ndarray:
                self.registered_arrays += [attr_name]
                print('   - ' + attr_name)

                
    def mask(self, mask):
        ''' Applying a mask on all the arrays of the dataset '''
        
        # For every numpy array in the class, we apply the mask
        for array_name in self.registered_arrays:
            masked_array = getattr(self, array_name)[mask]
            setattr(self, array_name, masked_array)
            
        self.N = self.ra.shape[0]
        
        
    def get_cartesian_coords(self, band='I'):
        ''' Transforms the ra, dec, dist coordinates in cartesian coordinates '''
        delta = self.dec * np.pi / 180.0 + np.pi * 0.5
        alpha = -self.ra  * np.pi / 180.0
        
        if band=='I':
            dist = self.d_I
        elif band=='V':
            dist = self.d_V
            
        x = np.sin(delta) * np.cos(alpha) * dist
        y = np.sin(delta) * np.sin(alpha) * dist
        z = np.cos(delta) * dist
        
        return np.stack((x, y, z)).T

        

# Now that we have a class we create datasets corresponding to the LMC and SMC Fundamental mode Cepheids :
lmc_fund = Dataset('data/LMC_Ceph_Fund.dat', compute_extinction=False)
smc_fund = Dataset('data/SMC_Ceph_Fund.dat', compute_extinction=False)

# We create some masks to remove some stars
# First we want only stars that belong to the LMC/SMC and not stragglers somewhere else, so
# we cut on the position on the sky, we take everythign with RA in [-25.0, 100.0] and Dec in [-80.0, -60.0]
lmc_pos_mask = (lmc_fund.ra > -25.0) & (lmc_fund.ra < 100.0) \
               & (lmc_fund.dec > -80.0) & (lmc_fund.dec < -60.0)
smc_pos_mask = (smc_fund.ra > -25.0) & (smc_fund.ra < 100.0) \
               & (smc_fund.dec > -80.0) & (smc_fund.dec < -60.0)

# We also remove everythign that is "too close" (<1 kpc) or "too far" (> 150 kpc)
lmc_dist_mask = (lmc_fund.d_I > 1.0) & (lmc_fund.d_I < 150.0)
smc_dist_mask = (smc_fund.d_I > 1.0) & (smc_fund.d_I < 150.0)

# And we mask
lmc_fund.mask(lmc_pos_mask & lmc_dist_mask)
smc_fund.mask(smc_pos_mask & smc_dist_mask)


# We can now, plot a few things
# We create a structure to plot a figure with 4 panels (2 rows, 2 cols)
fig, ax = plt.subplots(2, 2, figsize=(15, 15))

# Here we are going to plot on the top left the distance vs dec of the LMC on I band
ax[0][0].scatter(lmc_fund.d_I, lmc_fund.dec, s=1, alpha=0.5)
ax[0][0].set_xlabel('Distance (kpc)')
ax[0][0].set_ylabel('Declination (deg)')
ax[0][0].set_title('LMC I band')

# Now for the top right plot, same thing with the SMC
ax[0][1].scatter(smc_fund.d_I, smc_fund.dec, s=1, alpha=0.5)
ax[0][1].set_xlabel('Distance (kpc)')
ax[0][1].set_ylabel('Declination (deg)')
ax[0][1].set_title('SMC I band')

# Bottom left, LMC, V band
ax[1][0].scatter(lmc_fund.d_V, lmc_fund.dec, s=1, alpha=0.5)
ax[1][0].set_xlabel('Distance (kpc)')
ax[1][0].set_ylabel('Declination (deg)')
ax[1][0].set_title('LMC V band')

# Bottom right, SMC, V band
ax[1][1].scatter(smc_fund.d_V, smc_fund.dec, s=1, alpha=0.5)
ax[1][1].set_xlabel('Distance (kpc)')
ax[1][1].set_ylabel('Declination (deg)')
ax[1][1].set_title('SMC V band')

# And we display this
plt.show()


