#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 12:23:29 2025

@author: esivkova
"""

from astroquery.simbad import Simbad
from astroquery.gaia import Gaia
import matplotlib.pyplot as plt
from astropy.time import Time
import astropy.units as u
import pandas as pd
import numpy as np
import kepler
import os
import spleaf
import copy
import requests
import xml.etree.ElementTree as ET
from kepmodel.astro import AstroModel as AstrometricModel

def orbit2(theta, jd):
    theta = {'a': theta['a']*u.mas, 
             'i': theta['i']*u.deg, 
             'Omega': theta['Omega']*u.deg, 
             'e':theta['e'],
             'w':theta['w']*u.deg, 
             'T0': theta['T0'], 
             'P':theta['P']*u.day}
    P = theta['P']
    a = theta['a']
    e = theta['e']
    i = theta['i']
    Omega = theta['Omega']
    omega = theta['w']
    T = theta['T0']

    M = 2*np.pi/P*(jd-T)
    E, cosE, sinE = kepler.kepler(M.value, e)
    
    nu = 2*np.arctan2((1+e)**0.5*np.sin(E/2),(1-e)**0.5*np.cos(E/2))*u.rad
    r = a*(1-e**2)/(1+e*np.cos(nu)) 
    
    delt_ra = r*(np.sin(Omega)*np.cos(omega+nu) + np.cos(i)*np.cos(Omega)*np.sin(omega+nu)) 
    delt_dec = r*(np.cos(Omega)*np.cos(omega+nu) - np.cos(i)*np.sin(Omega)*np.sin(omega+nu))
    return np.array([delt_ra, delt_dec])

class SimBinary:
    def __init__(self, ObjectParameters, DataRelease = 4, SaveGost = True, GaiaPuls = True):
        
        # filtering nans
        ObjectParameters = {k: ObjectParameters[k] for k in ObjectParameters\
                            if not pd.isnull(ObjectParameters[k])}
        # checking that all parameters are ok
        self.check_params(ObjectParameters, DataRelease)
        # defining few useful parameters
        self.ObjectParameters = ObjectParameters
        self.ObjectName = ObjectParameters['Object']
        self.ObjectType = ObjectParameters['type']
        self.DataRelease = DataRelease
        self.SaveGost = SaveGost
        
        Trefs = {1:2015.0,
                 2:2015.5,
                 3:2016,
                 4:2017.5,
                 5:2020}
        self.Tref = Time(Trefs[DataRelease],format='decimalyear')
        self.has_pulsation = False
        self.r1 = 0 
        self.r2 = 1
        self.ra0 = 0*u.mas
        self.dec0 = 0*u.mas
        
        if 'ra0' in ObjectParameters:
            self.ra0 = ObjectParameters['ra0']*u.mas
        if 'dec0' in ObjectParameters:
            self.dec0 = ObjectParameters['dec0']*u.mas
        
        self.querySimbadGaia()
        
        gostdata = self.LoadGost()
            
        if self.ObjectType == 'binary':
            f_main = 1
            f_comp = 10**(-0.4*(ObjectParameters['Vcomp']-self.ObjectGmag))
            self.r1 = f_main/(f_comp+f_main) # main star
            self.r2 = f_comp/(f_comp+f_main) # companion
        
        if self.ObjectPMDEC is None and self.DataRelease>=3:
            print('Applying correction for DR3 proper motion...')
            
            self.LimitGost(gostdata, DR=3)
            self.CepheidCase(GaiaPuls)
            
            # in work
            
            self.ObjectPMRA, self.ObjectPMDEC = 0, 0
            w_bs = self.SimDR4()
            
            mA = np.array([
                np.sin(self.scanAngleRAD),                # alpha0
                self.reltimes*np.sin(self.scanAngleRAD),        # pmra
                np.cos(self.scanAngleRAD),                # delta0
                self.reltimes*np.cos(self.scanAngleRAD),        # pmdec
                self.prlFactorAL                          # parallax
                ]).T
            werr = np.array(len(w_bs)*[0.02])
            Cinv = np.diag(1/werr**2)
            p_fit = np.linalg.solve(mA.T @ Cinv @ mA, mA.T @ Cinv @ w_bs)
            _, pmra, _, pmdec, _ = p_fit
            print('Vector:', pmra*365.25, pmdec*365.25)
            self.ObjectPMRA = self.ObjectPMRA_DR3cat - pmra*365.25
            self.ObjectPMDEC = self.ObjectPMDEC_DR3cat - pmdec*365.25
            print(f'Proper motion corrected to: {self.ObjectPMRA} \
                  and {self.ObjectPMDEC} mas')
        elif self.ObjectPMDEC is None:
            self.ObjectPMRA=self.ObjectPMRA_DR3cat
            self.ObjectPMDEC=self.ObjectPMDEC_DR3cat
        
        self.LimitGost(gostdata, DR=self.DataRelease)
        self.CepheidCase(GaiaPuls)

                
        
    def check_params(self, ObjectParameters, DataRelease):
        
        if DataRelease not in [1, 2, 3, 4, 5]:
            raise ValueError("Please, choose on of the Gaia DR: 1, 2, 3, 4, 5. \
                             The current value '{DataRelease} is not supported.'")
        
        schema = {
             'Object': {'required': True, 'type': str},
             'type':   {'required': True, 'type': str, 'selection':['BH', 'binary', 'cepheid']},
             'ra':     {'required': False,'type': (float, int, np.floating), 'range': [0, 360]},
             'dec':    {'required': False,'type': (float, int, np.floating), 'range': [-90, 90]},
             'id3':    {'required': False,'type': str},
             'P':      {'required': True, 'type': (float, int, np.floating), 'range': [0, 10e6]},
             'a':      {'required': True, 'type': (float, int, np.floating), 'range': [0, 160]},
             'e':      {'required': True, 'type': (float, int, np.floating), 'range': [0, 1]},
             'i':      {'required': True, 'type': (float, int, np.floating), 'range': [0, 180]},
             'Omega':  {'required': True, 'type': (float, int, np.floating), 'range': [0, 360]},
             'w':      {'required': True, 'type': (float, int, np.floating), 'range': [0, 360]},
             'T0':     {'required': True, 'type': (float, int, np.floating)},
             'q':      {'required': True, 'type': (float, int, np.floating), 'range': [0, 10e3]},
             'pll':    {'required': True, 'type': (float, int, np.floating), 'range': [0, 10e3]},
             'vra':    {'required': False,'type': (float, int, np.floating)},
             'vdec':   {'required': False,'type': (float, int, np.floating)},
             'Ppuls':  {'required': False,'type': (float, int, np.floating), 'range': [0, 10e6]},
             'T0puls': {'required': False,'type': (float, int, np.floating)},
             'Vmax':   {'required': False,'type': (float, int, np.floating)},
             'Vmin':   {'required': False,'type': (float, int, np.floating)},
             'Vmain':  {'required': False,'type': (float, int, np.floating)},
             'Vcomp':  {'required': False,'type': (float, int, np.floating)},
             'fratio': {'required': False,'type': (float, int, np.floating)},
             'ra0':    {'required': False,'type': (float, int, np.floating)},
             'dec0':   {'required': False,'type': (float, int, np.floating)}
        }
        
        for key, rule in schema.items():
            if rule['required'] and key not in ObjectParameters:
                raise KeyError(f"The parameter '{key}' is missing. Please, add \
                               it to the parameters dictionary.")
               
            if key in ObjectParameters:
                expected = rule["type"]
                value = ObjectParameters[key]
                
                if not isinstance(value, expected):
                    raise TypeError(f"The parameter '{key}' is {type(ObjectParameters[key])}.\
                                    Should be {expected}.")
            
                if 'range' in rule:
                    min_v, max_v = rule['range']
                    if not (min_v <= value <= max_v):
                        raise ValueError(f"The parameter '{key}' is out of range \
                                         ({min_v}, {max_v}) with value {value}.")
                if 'selection' in rule:
                    if value not in rule['selection']:
                        raise ValueError(f"The object type '{value}' doesn\'t \
                                         correspond to the supported ones: {rule['selection']}")
    
    def querySimbadGaia(self):
        self.ObjectRA = None
        self.ObjectDEC = None
        self.id3 = None 
        self.ObjectPMRA = None
        self.ObjectPMDEC = None
        self.ObjectGmag = None
        
        if 'id3' in self.ObjectParameters:
            self.id3 = self.ObjectParameters['id3']
        
        if 'ra' in self.ObjectParameters and 'dec' in self.ObjectParameters:
            self.ObjectRA = self.ObjectParameters['ra']
            self.ObjectDEC = self.ObjectParameters['dec']
            
        if 'pmra' in self.ObjectParameters and 'pmdec' in self.ObjectParameters:
            self.ObjectPMRA = self.ObjectParameters['pmra']
            self.ObjectPMDEC = self.ObjectParameters['pmdec']
        
        if 'Vmain' in self.ObjectParameters:
            self.ObjectGmag = self.ObjectParameters['Vmain']
        
        if self.id3 is None:
            Simbad.add_votable_fields('ids')
            result = Simbad.query_object(self.ObjectName)
            
            if len(result) == 0:
                raise ValueError('The object was not resolved by Simbad. \
                    Try with to change the target name or to add RA, \
                        DEC, VRA, VDEC and GAIA DR3 ID to avoid Simbad query.')
            
            if self.id3 is None:
                ids = result['ids'][0].split('|')
                gaia_id = [s for s in ids if 'Gaia DR3' in s]
                if len(gaia_id) == 0:
                    raise ValueError('The object is not in DR3.')
                # Take only the number
                self.id3 = gaia_id[0][9:]
                print('Gaia DR3 ID added with Simbad')
                
        if None in [self.ObjectRA, self.ObjectPMRA, self.ObjectGmag]:
            Gaia.ROW_LIMIT = 1  
            query = f"""
            SELECT *
            FROM gaiadr3.gaia_source
            WHERE source_id = {self.id3}
            """
            job = Gaia.launch_job(query)
            object_data = job.get_results()
            
            if self.ObjectRA is None:
                self.ObjectRA = object_data['ra'].data[0]
                self.ObjectDEC = object_data['dec'].data[0]
                print('RA/DEC coordinates added with Gaia DR3')
            if self.ObjectPMDEC is None:
                self.ObjectPMRA_DR3cat = object_data['pmra'].data[0]
                self.ObjectPMDEC_DR3cat = object_data['pmdec'].data[0]
                print('Proper motion RA/DEC added with Gaia DR3')
            if self.ObjectGmag is None:
                self.ObjectGmag = object_data['phot_g_mean_mag']
                print('Gmag added with Gaia DR3')
            
    def LoadGost(self):
        
        name = self.ObjectName.replace(' ', '_')
        filepath = f"gost/gost_{name}.csv"
        if os.path.isfile(filepath):
            gostdata = pd.read_csv(filepath, sep=',')
        else:
            gostdata = self.DownloadGost(self.ObjectRA, self.ObjectDEC, self.ObjectName)
            if self.SaveGost:
                if not os.path.isdir('gost/'):
                    os.mkdir('gost/')
                print(f"Downloading GOST data to {os.getcwd()+'/gost'}.")
                gostdata.to_csv(filepath, sep =',')
        return gostdata
    
    def LimitGost(self, gostdata, DR):
        TstopDRs = {1:'2015-09-16T00:00:00',
                 2:'2016-05-23T11:35:00',
                 3:'2017-05-28T08:44:00',
                 4:'2020-01-20T22:00:00',
                 5:'2025-01-16T00:00:00'}
        Tstop = Time(TstopDRs[DR],format='isot')
        
        gostdata = gostdata[gostdata['ObservationalTimeBarycentre']<Tstop.jd]
        
        self.times = Time([Time(t, format='jd') \
                           for t in gostdata['ObservationalTimeBarycentre']])
        
        self.scanAngleRAD = gostdata['scanAngle']
        self.prlFactorAL = gostdata['parallaxFactorAL']
        self.prlFactorAC = gostdata['parallaxFactorAC']
        
        self.scanAngleDEG = np.rad2deg(self.scanAngleRAD)
        self.reltimes = (self.times - self.Tref).to(u.day)
        self.timesjd = self.times.to_value('jd')
        
    def DownloadGost(self, ra, dec, target_name):

        # Adapted from Download_HIP_Gaia_GOST by Yicheng Rui
        # https://github.com/ruiyicheng/Download_HIP_Gaia_GOST/tree/main
        url = f"https://gaia.esac.esa.int/gost/GostServlet?ra={str(ra)}+&dec={str(dec)}"

        with requests.Session() as s:
            s.get(url)
            headers = {"Cookie": f"JSESSIONID={s.cookies.get_dict()['JSESSIONID']}"}
            response = s.get(url, headers=headers, timeout=1000)#,proxies=proxies)
        root = ET.fromstring(response.text)
        columns = ["Target", "CcdRow", "scanAngle", "parallaxFactorAL", 
                   "parallaxFactorAC", "ObservationalTimeBarycentre"]
        rows = []
        name = root.find('./targets/target/name').text

        for event in root.findall('./targets/target/events/event'):
            details = event.find('details')
            ccdRow = details.find('ccdRow').text
            scanAngle = details.find('scanAngle').text
            parallaxFactorAl = details.find('parallaxFactorAl').text
            parallaxFactorAc = details.find('parallaxFactorAc').text
            observationTimeAtBarycentre = event.find('eventTcbBarycentricJulianDateAtBarycentre').text
            rows.append([name, ccdRow, scanAngle, parallaxFactorAl, parallaxFactorAc, observationTimeAtBarycentre])
        data = pd.DataFrame(rows, columns=columns)
        data = data.astype({"Target": str, "CcdRow": int,"scanAngle": float,"parallaxFactorAL": float,"parallaxFactorAC": float,"ObservationalTimeBarycentre": float })
        data['Target']=[target_name]*len(data)

        return data
    
    def CepheidCase(self, GaiaPuls):
        if self.ObjectType == 'cepheid' and not GaiaPuls:
            required = ['Vmin', 'Vmax', 'Vcomp', 'Ppuls', 'T0puls']
            if all(key in self.ObjectParameters for key in required):
                self.addPulsation()
            else:
                raise KeyError("Please, provide the next parameters to \
                               approximate Cepheid pulsation: Vmin, Vmax, Vcomp, Ppuls, T0puls")
        elif self.ObjectType == 'cepheid':
            self.addPulsationGaia()
    
    def addPulsation(self):
        Vmin = self.ObjectParameters['Vmin']
        Vmax = self.ObjectParameters['Vmax']
        Vcomp = self.ObjectParameters['Vcomp']
        Ppuls = self.ObjectParameters['Ppuls']
        T0 = self.ObjectParameters['T0puls']
        f_max = 1 # reference flux at Cepheid maximum
        # minimal cepheid flux and companion flux relative to ref flux
        # F/Fref = 10**(-0.4*(m-mref))
        f_min = 10**(-0.4*(Vmin-Vmax))
        f_comp = 10**(-0.4*(Vcomp-Vmax))

        f_ceph = (f_max-f_min)/2 *\
            np.cos(2*np.pi*(self.timesjd - T0)/Ppuls) \
                + (f_max+f_min)/2 #cepheid flux modelisation
        # flux fraction for each component        
        self.r1 = f_ceph/(f_comp+f_ceph) # cepheid
        self.r2 = f_comp/(f_comp+f_ceph) # companion
        
        # binary system without the pulsating component, non-pulsating system (nps)
        self.r1_nps = f_max/(f_comp+f_max) # cepheid
        self.r2_nps = f_comp/(f_comp+f_max) # companion
        
        self.has_pulsation = True
        
    def addPulsationGaia(self):
        
        Gaia.ROW_LIMIT = 1  
        query = f"""
        SELECT *
        FROM gaiadr3.vari_cepheid
        WHERE source_id = {str(self.id3)}
        """
        job = Gaia.launch_job(query)
        cepheid_data = job.get_results()
        
        if len(cepheid_data) == 0:
            raise ValueError('The object was not found in gaiadr3.vari_cepheid.\
                             You can add the pulsation parameters manually.')
        
        A0 = cepheid_data['int_average_g'].data[0]
        N = cepheid_data['num_harmonics_for_p1_g'].data[0] 
        As = cepheid_data['fund_freq1_harmonic_ampl_g'][0][:N].compressed()
        phis = cepheid_data['fund_freq1_harmonic_phase_g'][0][:N].compressed()
        # f1 = cepheid_data['fund_freq1'].data[0]
        T0 = cepheid_data['reference_time_g'].data[0]
        P = cepheid_data['pf'].data[0]
        if np.ma.is_masked(P): 
            P = cepheid_data['p1_o'].data[0]
            print('The p1_o period used')
        if np.ma.is_masked(P): 
            P = 1/cepheid_data['fund_freq1'].data[0]
            print('The 1/fund_freq period used')
        k = np.arange(1,N+1)
        arg = 2*np.pi*k[:, None]*(self.timesjd - T0)/P + phis[:, None]
        puls = A0 + np.sum(As[:, None]*np.cos(arg), axis=0)
        
        
        maxpuls = np.min(puls) # reference flux at Cepheid maximum
        # minimal cepheid flux and companion flux relative to ref flux
        # F/Fref = 10**(-0.4*(m-mref))
        f_ceph = 10**(-0.4*(puls-maxpuls))
        
        f_mean = np.mean(f_ceph)
        
        if 'Vcomp' in self.ObjectParameters:
            f_comp = 10**(-0.4*(self.ObjectParameters['Vcomp']-maxpuls))
        elif 'fratio' in self.ObjectParameters:
            f_comp = self.ObjectParameters['fratio']/100 * f_mean
        else:
            raise KeyError("Please add a lux ratio or the magnutude of the companion.")
        
        
        # flux fraction for each component        
        self.r1 = f_ceph/(f_comp+f_ceph) # cepheid
        self.r2 = f_comp/(f_comp+f_ceph) # companion
        
        # binary system without the pulsating component, non-pulsating system (nps)
        self.r1_nps = f_mean/(f_comp+f_mean) # cepheid
        self.r2_nps = f_comp/(f_comp+f_mean) # companion
        
        self.has_pulsation = True
        
    @staticmethod
    def orbit(theta, jd): # orbit model
        P = theta['P']
        a = theta['a']
        e = theta['e']
        i = theta['i']
        Omega = theta['Omega']
        omega = theta['w']
        T = theta['T0']

        M = 2*np.pi/P*(jd-T)
        E, cosE, sinE = kepler.kepler(M.value, e)
        
        nu = 2*np.arctan2((1+e)**0.5*np.sin(E/2),(1-e)**0.5*np.cos(E/2))*u.rad
        r = a*(1-e**2)/(1+e*np.cos(nu)) 
        
        delt_ra = r*(np.sin(Omega)*np.cos(omega+nu) + np.cos(i)*np.cos(Omega)*np.sin(omega+nu)) 
        delt_dec = r*(np.cos(Omega)*np.cos(omega+nu) - np.cos(i)*np.sin(Omega)*np.sin(omega+nu))
        return np.array([delt_ra, delt_dec])
    
    def SimDR4(self):
        # primary star/BH parameters, w+pi because it is on the opposite side to companion
        params1 = {'a': self.ObjectParameters['a']*u.mas*self.ObjectParameters['q']/(1+self.ObjectParameters['q']), 
                   'i': self.ObjectParameters['i']*u.deg, 
                   'Omega': self.ObjectParameters['Omega']*u.deg, 
                   'e':self.ObjectParameters['e'],
                   'w':(self.ObjectParameters['w']+180)*u.deg, 
                   'T0': (self.ObjectParameters['T0']-self.Tref.jd)*u.day, 
                   'P':self.ObjectParameters['P']*u.day}
        # companion parameters, q=1 in case of BH because the orbit is already photocentric
        q_comp = 1
        if self.ObjectType != 'BH': 
            q_comp = 1/(1+self.ObjectParameters['q'])
        params2 = {'a': self.ObjectParameters['a']*u.mas*q_comp, 
                   'i': self.ObjectParameters['i']*u.deg, 
                   'Omega': self.ObjectParameters['Omega']*u.deg, 
                   'e':self.ObjectParameters['e'],
                   'w':(self.ObjectParameters['w'])*u.deg, 
                   'T0': (self.ObjectParameters['T0']-self.Tref.jd)*u.day, 
                   'P':self.ObjectParameters['P']*u.day}
        
        vra = self.ObjectPMRA*u.mas/u.year 
        vdec = self.ObjectPMDEC*u.mas/u.year
        pll = self.ObjectParameters['pll']
        
        self.ra1, self.dec1 = self.orbit(params1, self.reltimes)
        self.ra2, self.dec2 = self.orbit(params2, self.reltimes)
        
        if self.ObjectType != 'BH':
            a_ph = (self.ObjectParameters['q']/(1+self.ObjectParameters['q'])*np.mean(self.r1) -\
                    1/(1+self.ObjectParameters['q'])*np.mean(self.r2))*self.ObjectParameters['a']
            self.params_ph = {'a': a_ph*u.mas, 
                       'i': self.ObjectParameters['i']*u.deg, 
                       'Omega': self.ObjectParameters['Omega']*u.deg, 
                       'e':self.ObjectParameters['e'],
                       'w':(self.ObjectParameters['w']+180)*u.deg, 
                       'T0': (self.ObjectParameters['T0']-self.Tref.jd)*u.day, 
                       'P':self.ObjectParameters['P']*u.day}
        else:
            self.params_ph = params2
            
        # photocenter position 
        self.ra_ph = self.ra1*self.r1 + self.ra2*self.r2
        self.dec_ph = self.dec1*self.r1 + self.dec2*self.r2

        # adding proper motion to the binary system (bs)
        self.ra_bs = self.ra_ph*u.mas + vra*self.reltimes
        self.dec_bs = self.dec_ph*u.mas + vdec*self.reltimes

        # proper motion alone to model single star (ss)
        self.ra_ss = (vra*self.reltimes).to(u.mas)
        self.dec_ss = (vdec*self.reltimes).to(u.mas)
        
        if self.has_pulsation:
            self.ra_nps = (self.ra1*self.r1_nps + self.ra2*self.r2_nps)*u.mas + vra*self.reltimes
            self.dec_nps = (self.dec1*self.r1_nps + self.dec2*self.r2_nps)*u.mas + vdec*self.reltimes
            
        # projecting parallax factors to ra, dec
        self.factra = -self.prlFactorAL*np.sin(self.scanAngleRAD)+self.prlFactorAC*np.cos(self.scanAngleRAD)
        self.factdec = self.prlFactorAL*np.cos(self.scanAngleRAD)+self.prlFactorAC*np.sin(self.scanAngleRAD)
        
        # adding projected parallax motion for visualisation
        self.ra_bs_pll = self.ra_bs.value+pll*self.factra
        self.dec_bs_pll = self.dec_bs.value+pll*self.factdec
        
        self.ra_ss_pll = self.ra_ss.value+pll*self.factra
        self.dec_ss_pll = self.dec_ss.value+pll*self.factdec
        
        if self.has_pulsation:
            self.ra_nps_pll = self.ra_nps.value+pll*self.factra
            self.dec_nps_pll = self.dec_nps.value+pll*self.factdec
        
        # ra1r, dec1r = self.orbit(params1, [0]*u.day)
        # ra2r, dec2r = self.orbit(params2, [0]*u.day)
        # ra_phr = ra1r*self.r1 + ra2r*self.r2
        # dec_phr = dec1r*self.r1 + dec2r*self.r2
        # ra_bsr = ra_phr +pll*-0.2182321618812346
        # dec_bsr = dec_phr + pll*0.9946392003384911
        # print('positions:', ra_bsr, dec_bsr)
        # self.ra0 = -ra_bsr*u.mas
        # self.dec0 = -dec_bsr*u.mas
        
        # gaia along scan projection for BS
        self.w_bs = (self.dec0 + self.dec_bs)*np.cos(self.scanAngleRAD) + (self.ra0 + self.ra_bs)*np.sin(self.scanAngleRAD) + pll*self.prlFactorAL
        
        self.w_ss = (self.dec0 + self.dec_ss)*np.cos(self.scanAngleRAD) + (self.ra0 + self.ra_ss)*np.sin(self.scanAngleRAD) + pll*self.prlFactorAL
        
        return self.w_bs
    
    def PlotSim(self, plot_dir=None):
        
        if self.has_pulsation:
            label1 = 'Cepheid'
            label2 = 'Companion'
            lw = 1
        elif self.ObjectType=='binary':
            label1 = 'Star 1'
            label2 = 'Star 2'
            lw = 1
        else:
            label1 = 'Black hole'
            label2 = 'Star'
            lw = 5
            
        fig, axs = plt.subplots(1,3, figsize=(15, 5), constrained_layout=True)
        fig.suptitle(self.ObjectName, fontsize=16)

        axs[0].set_title('Orbit motion')
        axs[0].plot(self.ra1, self.dec1, label=label1, marker='.', color = 'pink')
        axs[0].plot(self.ra2, self.dec2, label=label2, marker='.', color = 'lightskyblue', lw = lw)
        axs[0].plot(self.ra_ph, self.dec_ph, label='Photocentre', marker='.', color = 'black')
        axs[0].xaxis.set_inverted(True)
        axs[0].set_xlabel(r'$\Delta \alpha cos(\delta)$ [mas]')
        axs[0].set_ylabel(r'$\Delta \delta$ [mas]')
        axs[0].legend()
        axs[0].set_aspect('equal', adjustable='datalim')

        axs[1].set_title('Orbit + proper motions')
        axs[1].plot(self.ra_ss, self.dec_ss, label='Single star model', marker='.', color = 'peachpuff')
        if self.has_pulsation:
            axs[1].plot(self.ra_nps, self.dec_nps, label='Non pulsating system', marker='.', color='orange')
        axs[1].plot(self.ra_bs, self.dec_bs, label='Photocentre of the system', marker='.', color = 'black')
        axs[1].xaxis.set_inverted(True)
        axs[1].set_xlabel(r'$\Delta \alpha cos(\delta)$ [mas]')
        axs[1].set_ylabel(r'$\Delta \delta$ [mas]')
        axs[1].legend()
        # axs[1].set_aspect('equal', adjustable='datalim')

        # adding projected parallax motion for visualisation
        axs[2].set_title('On sky (orbit + proper + parallax motions)')
        axs[2].plot(self.ra_ss_pll, self.dec_ss_pll, 
                    label='Single star model', marker='.', color = 'peachpuff')
        if self.has_pulsation:
            axs[2].plot(self.ra_nps_pll, self.dec_nps_pll, 
                        label='Non pulsating system', marker='.', color='orange')
        axs[2].plot(self.ra_bs_pll, self.dec_bs_pll, 
                    label='Photocentre of the system', marker='.', color = 'black')
        axs[2].xaxis.set_inverted(True)
        axs[2].set_xlabel(r'$\Delta \alpha cos(\delta)$ [mas]')
        axs[2].set_ylabel(r'$\Delta \delta$ [mas]')
        axs[2].legend()
        # axs[2].set_aspect('equal', adjustable='datalim')
        
        if plot_dir is not None:
            fig.savefig(plot_dir+f'astrometry_{self.ObjectName}.png', 
                        dpi=300, bbox_inches="tight")
            
    def PlotSim2(self, plot_dir=None):
        
        if self.has_pulsation:
            label1 = 'Cepheid'
            label2 = 'Companion'
            lw = 1
        elif self.ObjectType=='binary':
            label1 = 'Star 1'
            label2 = 'Star 2'
            lw = 1
        else:
            label1 = 'Black hole'
            label2 = 'Star'
            lw = 5
            
        fig, axs = plt.subplots(1,2, figsize=(14, 7), constrained_layout=True)
        fig.suptitle(self.ObjectName.replace('_', ' '), fontsize=16)

        axs[0].set_title('Orbit motion')
        axs[0].plot(self.ra1, self.dec1, label=label1, marker='.', color = 'pink')
        axs[0].plot(self.ra2, self.dec2, label=label2, marker='.', color = 'lightskyblue', lw = lw)
        axs[0].plot(self.ra_ph, self.dec_ph, label='Photocentre', marker='.', color = 'black')
        axs[0].xaxis.set_inverted(True)
        axs[0].set_xlabel(r'$\Delta \alpha cos(\delta)$ [mas]')
        axs[0].set_ylabel(r'$\Delta \delta$ [mas]')
        axs[0].legend()
        axs[0].set_aspect('equal', adjustable='datalim')

        # adding projected parallax motion for visualisation
        axs[1].set_title('On sky (orbit + proper + parallax motions)')
        axs[1].plot(self.ra_ss_pll, self.dec_ss_pll, 
                    label='Single star model', marker='.', color = 'peachpuff')
        if self.has_pulsation:
            axs[1].plot(self.ra_nps_pll, self.dec_nps_pll, 
                        label='Non pulsating system', marker='.', color='orange')
        axs[1].plot(self.ra_bs_pll, self.dec_bs_pll, 
                    label='Photocentre of the system', marker='.', color = 'black')
        axs[1].xaxis.set_inverted(True)
        axs[1].set_xlabel(r'$\Delta \alpha cos(\delta)$ [mas]')
        axs[1].set_ylabel(r'$\Delta \delta$ [mas]')
        axs[1].legend()
        # axs[2].set_aspect('equal', adjustable='datalim')
        
        if plot_dir is not None:
            fig.savefig(plot_dir+f'astrometry_gaia_{self.ObjectName}.png', 
                        dpi=300, bbox_inches="tight")
            
    def get_dataframe(self, data_dir=None):
        # resulting datatframe
        sim_astrometry = pd.DataFrame()
        sim_astrometry['transit_id'] = np.arange(1,len(self.timesjd)+1,1) # we don't have the true ones
        sim_astrometry['ccd_id'] = 1 # because we simulate only one ccd
        sim_astrometry['obs_time_tcb'] = self.timesjd
        sim_astrometry['centroid_pos_al'] = self.w_bs
        sim_astrometry['centroid_pos_error_al'] = 0.02 # au pif
        sim_astrometry['parallax_factor_al'] = self.prlFactorAL
        sim_astrometry['scan_pos_angle'] = self.scanAngleDEG # BH3 notebook is made for degrees
        sim_astrometry['outlier_flag'] = 0
        
        if data_dir is not None:
            sim_astrometry.to_csv(data_dir+f'sim{self.ObjectName}.dat', 
                                  sep=' ', header=False, index=False)
        return sim_astrometry
        
class notebookDR4:
    def __init__(self, object_name, id3):
        #self.object_name
        self.object_name = object_name
        self.DR4_REFERENCE_EPOCH = Time('2017.5', format='jyear', scale='tcb')
        self.id3 = id3
        
    def load_file(self, filename_gaia_astro):
        columns = 'transitid ccd_id obs_time_tcb centroid_pos_al centroid_pos_error_al \
            parallax_factor_al scan_pos_angle outlier_flag'.split()
        self.gaiaastro = pd.read_csv(filename_gaia_astro, delim_whitespace=True, names=columns, comment='#')
        
    def load_dataframe(self, dataframe):
        columns = 'transitid ccd_id obs_time_tcb centroid_pos_al centroid_pos_error_al \
            parallax_factor_al scan_pos_angle outlier_flag'.split()
        dataframe.columns = columns
        self.gaiaastro = dataframe
        
        
    def fitthething(self, data_dir=None):
        
        source_id = self.id3
        
        gaia_astrometry = self.gaiaastro
        
        # filter our unused data
        gaia_astrometry = gaia_astrometry[gaia_astrometry['outlier_flag']!=1]

        # set auxiliary fields
        gaia_astrometry['source_id']  = source_id
        gaia_astrometry['relative_time_year'] = Time(gaia_astrometry['obs_time_tcb'], format='jd', scale='tcb').jyear - self.DR4_REFERENCE_EPOCH.jyear
        gaia_astrometry['mjd'] = Time(gaia_astrometry['obs_time_tcb'], format='jd', scale='tcb').mjd
        gaia_astrometry['relative_time_day'] = gaia_astrometry['relative_time_year'] * u.year.to(u.day)
        gaia_astrometry['cpsi_obs'] = np.cos(np.deg2rad(gaia_astrometry['scan_pos_angle']))
        gaia_astrometry['spsi_obs'] = np.sin(np.deg2rad(gaia_astrometry['scan_pos_angle']))
        
        include_jitter_term = False

        if include_jitter_term:
            astrometric_jitter_value = 0.05
        else:
            astrometric_jitter_value = 0.0
            
        # set up the single-star model with an additional jitter term of 0.05 mas
        single_star_model = AstrometricModel(gaia_astrometry['relative_time_day'].values, 
                                             gaia_astrometry['centroid_pos_al'].values, 
                                             gaia_astrometry['cpsi_obs'].values, 
                                             gaia_astrometry['spsi_obs'].values, 
                                             err=spleaf.term.Error(gaia_astrometry['centroid_pos_error_al'].values),
                                             jit=spleaf.term.Jitter(astrometric_jitter_value))
            
        # define the linear parameters
        single_star_model.add_lin(gaia_astrometry['spsi_obs'].values, 'ra')
        single_star_model.add_lin(gaia_astrometry['cpsi_obs'].values, 'dec')
        single_star_model.add_lin(gaia_astrometry['parallax_factor_al'].values, 'parallax')
        single_star_model.add_lin(gaia_astrometry['relative_time_year'].values * gaia_astrometry['spsi_obs'].values, 'mura')
        single_star_model.add_lin(gaia_astrometry['relative_time_year'].values * gaia_astrometry['cpsi_obs'].values, 'mudec')
        gaia_astrometry['ppfact_obs'] = gaia_astrometry['parallax_factor_al']
        gaia_astrometry['da_mas'] = gaia_astrometry['centroid_pos_al']
        gaia_astrometry['sigma_da_mas'] = gaia_astrometry['centroid_pos_error_al']

        # add jitter term
        if include_jitter_term:
            single_star_model.fit_param += ['cov.jit.sig']

        # perform the fit
        single_star_model.fit()
        
        # params_ss = single_star_model.get_param()
        print('SS model')
        single_star_model.show_param()
        
        model = copy.deepcopy(single_star_model)

        # Periodogram settings 
        Pmin = 5
        Pmax = 10000
        nfreq = 10000
        nu0 = 2 * np.pi / Pmax
        dnu = (2 * np.pi / Pmin - nu0) / (nfreq - 1)

        # compute periodogram
        nu, power = model.periodogram(nu0, dnu, nfreq)

        # convert from angular frequency to period
        P = 2 * np.pi / nu

        # identify highest peak and compute false-alarm probability (FAP)
        kmax = np.argmax(power)
        # faplvl = model.fap(power[kmax], nu.max())
        
        keplerian_model = copy.deepcopy(model)
        keplerian_model.add_keplerian_from_period(P[kmax])
        keplerian_model.fit()
        
        # params1 = keplerian_model.get_param()
        param = ['P', 'Tp', 'as', 'e', 'w', 'i', 'bigw']
        keplerian_model.set_keplerian_param('0', param=param)
        # params2 = keplerian_model.get_param()
        
        keplerian_parameters = {}
        for i, key in enumerate(keplerian_model.keplerian['0']._param):
            keplerian_parameters[key] = keplerian_model.keplerian['0']._par[i]

        linear_parameters = {}
        for i, key in enumerate(keplerian_model._lin_name):
            linear_parameters[key] = keplerian_model._lin_par[i]

        # a_m = convert_from_angular_to_linear(keplerian_parameters['as'], linear_parameters['parallax'])
        
        keplerian_parameters = {'a': keplerian_parameters['as'], 
                   'i': np.rad2deg(keplerian_parameters['i']), 
                   'Omega': np.rad2deg(keplerian_parameters['bigw']), 
                   'e':keplerian_parameters['e'],
                   'w':np.rad2deg(keplerian_parameters['w']), 
                   'T0': keplerian_parameters['Tp'], 
                   'P':keplerian_parameters['P']}
        
        if data_dir is not None:
            np.savetxt(f"{data_dir}/{self.object_name}_data.txt", np.array(list(keplerian_parameters.items())), fmt="%s")
        
        # print(f"Best-fit parameter (Campbell elements)\nkep.0.P is the period in days, kep.0.as is the semimajor axis in milli-arcseconds\n")
        print('BS model')
        keplerian_model.show_param()
        
        self.keplerian_model = keplerian_model
        self.linear_parameters = linear_parameters
        
        return keplerian_parameters
        