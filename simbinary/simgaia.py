#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 12:54:17 2026

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
from scipy.optimize import least_squares
from matplotlib.gridspec import GridSpec


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
        self.GaiaPuls = GaiaPuls
        
        Trefs = {1:2015.0,
                 2:2015.5,
                 3:2016,
                 4:2017.5,
                 5:2020}
        self.Tref = Time(Trefs[DataRelease],format='decimalyear')
        self.has_pulsation = False
        self.ra0 = 0
        self.dec0 = 0
        
        if 'ra0' in ObjectParameters:
            self.ra0 = ObjectParameters['ra0']
        if 'dec0' in ObjectParameters:
            self.dec0 = ObjectParameters['dec0']
        
        self.querySimbadGaia()
        
        gostdata = self.LoadGost()
        
        self.PrintInfo = True
        
        if self.ObjectPMDEC is None and self.ObjectPMDEC is None: #add PMRA condition
            print('Applying correction for DR3 proper motion...')
            
            self.LimitGost(gostdata, DR=3)
            
            # in work
            
            self.ObjectPMRA, self.ObjectPMDEC = 0, 0
            w_bs = self.SimWAL()
            
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
            print(f'Proper motion corrected to: {self.ObjectPMRA} and {self.ObjectPMDEC} mas')
        elif self.ObjectPMDEC is None:
            self.ObjectPMRA=self.ObjectPMRA_DR3cat
            self.ObjectPMDEC=self.ObjectPMDEC_DR3cat
        
        self.LimitGost(gostdata, DR=self.DataRelease)
        
        w_bs = self.SimWAL()
                
        
    def check_params(self, ObjectParameters, DataRelease):
        
        if DataRelease not in [1, 2, 3, 4, 5]:
            raise ValueError("Please, choose on of the Gaia DR: 1, 2, 3, 4, 5. \
                             The current value '{DataRelease} is not supported.'")
        
        schema = {
             'Object': {'required': True, 'type': str},
             'type':   {'required': True, 'type': str, 'selection':['BH', 'binary', 'cepheid', 'exoplanet']},
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
             'plx':    {'required': True, 'type': (float, int, np.floating), 'range': [0, 10e3]},
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
        self.reltimes = (self.times - self.Tref).to(u.day).value
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
 
    def FluxRatio(self, times, Tplot=0):
        
        fdata = pd.DataFrame(columns=['r1', 'r2'])
        
        if self.ObjectType == 'cepheid' and self.GaiaPuls:
            puls, r1, r2, r1_nps, r2_nps = self.addPulsationGaia(times, Tplot)
            fdata['puls'] = puls
            fdata['r1'] = r1
            fdata['r2'] = r2
            fdata['r1_nps'] = r1_nps
            fdata['r2_nps'] = r2_nps
            self.has_pulsation = True
            
        elif self.ObjectType == 'cepheid':
            required = ['Vmin', 'Vmax', 'Vcomp', 'Ppuls', 'T0puls']
            if all(key in self.ObjectParameters for key in required):
                puls, r1, r2, r1_nps, r2_nps = self.addPulsation(times, Tplot)
                fdata['puls'] = puls
                fdata['r1'] = r1
                fdata['r2'] = r2
                fdata['r1_nps'] = r1_nps
                fdata['r2_nps'] = r2_nps
            else:
                raise KeyError("Please, provide the next parameters to \
                               approximate Cepheid pulsation: Vmin, Vmax, Vcomp, Ppuls, T0puls")
            self.has_pulsation = True
            
        elif self.ObjectType == 'binary':
            mag_main = - 2.5*np.log10(10**(-0.4*(self.ObjectGmag))-10**(-0.4*(self.ObjectParameters['Vcomp'])))
            f_main = 10**(-0.4*(mag_main-self.ObjectGmag))
            f_comp = 10**(-0.4*(self.ObjectParameters['Vcomp']-self.ObjectGmag))
            fdata['r1'] = [f_main/(f_comp+f_main)] # main star
            fdata['r2'] = [f_comp/(f_comp+f_main)] # companion
            
        elif self.ObjectType == 'BH':
            fdata['r1'] = [1] # Star
            fdata['r2'] = [0] # BH
            
        elif self.ObjectType == 'exoplanet':
            fdata['r1'] = [1] # host star
            fdata['r2'] = [0] # exoplanet
            
        else:
            raise KeyError(f"Please, select between: binary, cepheod, BH, exoplanet.")
        
        return fdata
        
    def addPulsation(self, times, Tplot):
            
        Vmin = self.ObjectParameters['Vmin']
        Vmax = self.ObjectParameters['Vmax']
        Vcomp = self.ObjectParameters['Vcomp']
        Ppuls = self.ObjectParameters['Ppuls']
        T0 = self.ObjectParameters['T0puls']
        T0 = T0 - self.Tref.jd # converting T0 with a correct time reference
        T0 = T0 - Tplot
        print(Vmin, Vmax, Vcomp, Ppuls, T0)
        # minimal cepheid flux and companion flux relative to ref flux
        # F/Fref = 10**(-0.4*(m-mref))
        Vmean = np.mean([Vmax, Vmin]) # reference flux 
        f_comp = 10**(-0.4*(Vcomp-Vmean))
        
        puls = (Vmax-Vmin)/2 *\
            np.cos(2*np.pi*(times - T0)/Ppuls) \
                + (Vmax+Vmin)/2
        
        puls_ceph = - 2.5*np.log10(10**(-0.4*(Vmean))-10**(-0.4*(Vcomp)))
        f_ceph = 10**(-0.4*(puls_ceph-Vmean))
        # flux fraction for each component        
        r1 = f_ceph/(f_comp+f_ceph) # cepheid
        r2 = f_comp/(f_comp+f_ceph) # companion
        
        # binary system without the pulsating component, non-pulsating system (nps)
        # f_mean = np.mean([f_max, f_min])
        f_mean = 10**(-0.4*(Vmean-Vmax))*np.ones(len(puls))
        # f_mean = np.mean(f_ceph)
        r1_nps = f_mean/(f_comp+f_mean) # cepheid
        r2_nps = f_comp/(f_comp+f_mean) # companion
        
        return puls, r1, r2, r1_nps, r2_nps
    
    def addPulsationGaia(self, times, Tplot):
        
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
        T0 = T0 + Time(2016,format='decimalyear').jd # converting T0 to jd, 2016 DR3 ref
        T0 = T0 - self.Tref.jd # converting T0 with a correct time reference
        T0 = T0 - Tplot
        P = cepheid_data['pf'].data[0]
        if np.ma.is_masked(P): 
            P = cepheid_data['p1_o'].data[0]
            if self.PrintInfo:   
                print('The p1_o period used')
        if np.ma.is_masked(P): 
            P = 1/cepheid_data['fund_freq1'].data[0]
            if self.PrintInfo:   
                print('The 1/fund_freq period used')
        k = np.arange(1,N+1)
        arg = 2*np.pi*k[:, None]*(times - T0)/P + phis[:, None]
        puls = A0 + np.sum(As[:, None]*np.cos(arg), axis=0)
        
        self.ObjectParameters['Ppuls'] = P
        
        puls_ceph = - 2.5*np.log10(10**(-0.4*(puls))-10**(-0.4*(self.ObjectParameters['Vcomp'])))
        f_ceph = 10**(-0.4*(puls_ceph-self.ObjectGmag))
        f_comp = 10**(-0.4*(self.ObjectParameters['Vcomp']-self.ObjectGmag))
        f_mean = np.mean(f_ceph)*np.ones(len(puls))
        
        # flux fraction for each component  
        r1 = f_ceph/(f_comp+f_ceph) # cepheid
        r2 = f_comp/(f_comp+f_ceph) # companion
        
        # binary system without the pulsating component, non-pulsating system (nps)
        r1_nps = f_mean/(f_comp+f_mean) # cepheid
        r2_nps = f_comp/(f_comp+f_mean) # companion

        return puls, r1, r2, r1_nps, r2_nps

    def orbit(self, theta, times): # orbit model
        
        to_rad = (u.deg).to(u.rad)
    
        P = theta['P']
        a = theta['a']
        e = theta['e']
        i = theta['i']*to_rad
        Omega = theta['Omega']*to_rad
        omega = theta['w']*to_rad
        T = theta['T0']

        M = 2*np.pi/P*(times-T)
        E, cosE, sinE = kepler.kepler(M, e)
        
        nu = 2*np.arctan2((1+e)**0.5*np.sin(E/2),(1-e)**0.5*np.cos(E/2))
        r = a*(1-e**2)/(1+e*np.cos(nu)) 
        
        delt_ra = r*(np.sin(Omega)*np.cos(omega+nu) + np.cos(i)*np.cos(Omega)*np.sin(omega+nu)) 
        delt_dec = r*(np.cos(Omega)*np.cos(omega+nu) - np.cos(i)*np.sin(Omega)*np.sin(omega+nu))
        return np.array([delt_ra, delt_dec])

    def SimGaia(self, times, fdata, factra, factdec):
        # primary star/BH parameters, w+pi because it is on the opposite side to companion
        
        q_comp = self.ObjectParameters['q']/(1+self.ObjectParameters['q'])
        ang1 = 180
        ang2 = 0
        if self.ObjectType == 'BH' or self.ObjectType == 'exoplanet': 
            # companion parameters, q=1 in case of BH because the orbit is already photocentric
            q_comp = 1
            ang1 = 0
            ang2 = 180
            
        params1 = {'a': self.ObjectParameters['a']*q_comp, 
                   'i': self.ObjectParameters['i'], 
                   'Omega': self.ObjectParameters['Omega'], 
                   'e':self.ObjectParameters['e'],
                   'w':(self.ObjectParameters['w']+ang1), 
                   'T0': ((self.ObjectParameters['T0']-self.Tref.jd)*u.day).value, 
                   'P':self.ObjectParameters['P']}
        
        params2 = {'a': self.ObjectParameters['a']*1/(1+self.ObjectParameters['q']), 
                   'i': self.ObjectParameters['i'], 
                   'Omega': self.ObjectParameters['Omega'], 
                   'e':self.ObjectParameters['e'],
                   'w':(self.ObjectParameters['w']+ang2), 
                   'T0': ((self.ObjectParameters['T0']-self.Tref.jd)*u.day).value, 
                   'P':self.ObjectParameters['P']}
        
        # convert pm to mas/day
        vra = self.ObjectPMRA/365.25
        vdec = self.ObjectPMDEC/365.25
        plx = self.ObjectParameters['plx']
        
        data = pd.DataFrame(columns=['ra1', 'dec1', 'ra2', 'dec2', 
                                     'ra_ph', 'dec_ph', 'ra_bs', 'dec_bs',
                                     'ra_ss', 'dec_ss', 'ra_nps', 'dec_nps'
                                     'factra', 'factdec', 'ra_bs_plx', 'dec_bs_plx',
                                     'ra_ss_plx', 'dec_ss_plx', 'ra_nps_plx', 'dec_nps_plx',
                                     'w_bs', 'w_ss', 'w_nps'])
        
        data['ra1'], data['dec1'] = self.orbit(params1, times)
        data['ra2'], data['dec2'] = self.orbit(params2, times)
        
        
        if self.ObjectType != 'BH' or self.ObjectType == 'exoplanet':
            a_ph = (self.ObjectParameters['q']/(1+self.ObjectParameters['q'])*np.mean(fdata['r1']) -\
                    1/(1+self.ObjectParameters['q'])*np.mean(fdata['r2']))*self.ObjectParameters['a']
            self.params_ph = {'a': a_ph, 
                       'i': self.ObjectParameters['i'], 
                       'Omega': self.ObjectParameters['Omega'], 
                       'e':self.ObjectParameters['e'],
                       'w':(self.ObjectParameters['w']+180), 
                       'T0': ((self.ObjectParameters['T0']-self.Tref.jd)*u.day).value, 
                       'P':self.ObjectParameters['P']}
        else:
            self.params_ph = params1
            
        # photocenter position 
        data['ra_ph'] = data['ra1']*fdata['r1'].values + data['ra2']*fdata['r2'].values
        data['dec_ph'] = data['dec1']*fdata['r1'].values + data['dec2']*fdata['r2'].values
        
        
        # adding proper motion to the binary system (bs)
        data['ra_bs'] = data['ra_ph'] + vra*times
        data['dec_bs'] = data['dec_ph'] + vdec*times

        # proper motion alone to model single star (ss)
        data['ra_ss'] = vra*times
        data['dec_ss'] = vdec*times
        
        if self.has_pulsation:
            data['ra_ph_nps'] = (data['ra1']*fdata['r1_nps'] + data['ra2']*fdata['r2_nps'])
            data['dec_ph_nps'] = (data['dec1']*fdata['r1_nps'] + data['dec2']*fdata['r2_nps'])
            data['ra_nps'] = (data['ra1']*fdata['r1_nps'] + data['ra2']*fdata['r2_nps']) + vra*times
            data['dec_nps'] = (data['dec1']*fdata['r1_nps'] + data['dec2']*fdata['r2_nps']) + vdec*times
            self.a_ph_min = (self.ObjectParameters['q']/(1+self.ObjectParameters['q'])*np.min(fdata['r1']) -\
                    1/(1+self.ObjectParameters['q'])*np.max(fdata['r2']))*self.ObjectParameters['a']
            self.a_ph_max = (self.ObjectParameters['q']/(1+self.ObjectParameters['q'])*np.max(fdata['r1']) -\
                    1/(1+self.ObjectParameters['q'])*np.min(fdata['r2']))*self.ObjectParameters['a']
        
        # adding projected parallax motion for visualisation
        data['ra_bs_plx'] = data['ra_bs']+plx*factra
        data['dec_bs_plx'] = data['dec_bs']+plx*factdec
        
        data['ra_ss_plx'] = data['ra_ss']+plx*factra
        data['dec_ss_plx'] = data['dec_ss']+plx*factdec
        
        if self.has_pulsation:
            data['ra_nps_plx'] = data['ra_nps']+plx*factra
            data['dec_nps_plx'] = data['dec_nps']+plx*factdec
            
        
        return data
    
    def SimWAL(self):
        
        
        fdata = self.FluxRatio(self.reltimes)
        
        # projecting parallax factors to ra, dec
        factra = -self.prlFactorAL*np.sin(self.scanAngleRAD)+self.prlFactorAC*np.cos(self.scanAngleRAD)
        factdec = self.prlFactorAL*np.cos(self.scanAngleRAD)+self.prlFactorAC*np.sin(self.scanAngleRAD)
        
        data = self.SimGaia(self.reltimes, fdata, factra, factdec)
        self.Data = data
        
        self.w_bs = (self.dec0 + data['dec_bs'])*np.cos(self.scanAngleRAD) \
            + (self.ra0 + data['ra_bs'])*np.sin(self.scanAngleRAD) \
                + self.ObjectParameters['plx']*self.prlFactorAL

        self.w_ss = (self.dec0 + data['dec_ss'])*np.cos(self.scanAngleRAD) \
            + (self.ra0 + data['ra_ss'])*np.sin(self.scanAngleRAD) \
                + self.ObjectParameters['plx']*self.prlFactorAL
        
        if self.has_pulsation:
            self.w_nps = (self.dec0 + data['dec_nps'])*np.cos(self.scanAngleRAD) \
                + (self.ra0 + data['ra_nps'])*np.sin(self.scanAngleRAD) \
                    + self.ObjectParameters['plx']*self.prlFactorAL
        
        return self.w_bs
    
    def orbitTI(self, x, t):
        P, e, A, F, B, G, T = x

        M = 2*np.pi/P*(t-T)
        E, cosE, sinE = kepler.kepler(M, e)
        
        X = cosE - e
        Y = np.sqrt(1-e**2)*sinE
        
        delt_ra = A*X + F*Y
        delt_dec = B*X + G*Y

        return np.array([delt_ra, delt_dec])
    
    def residuals(self, x, t, y):
        return ((self.orbitTI(x, t)-y)**2).ravel()
    
    def SimPlot(self, times):
        
        fdata = self.FluxRatio(times)
        
        factra = -self.prlFactorAL*np.sin(self.scanAngleRAD)+self.prlFactorAC*np.cos(self.scanAngleRAD)
        factdec = self.prlFactorAL*np.cos(self.scanAngleRAD)+self.prlFactorAC*np.sin(self.scanAngleRAD)
        
        factors = np.array([factra, factdec])

        lower = [350, 0, -100, -100, -100, -100, -365]
        upper = [370, 1, 100, 100, 100, 100, 365]

        res_fit = least_squares(self.residuals, [369, 0.5, 1, 1, 1, 1, 0], args=(self.reltimes, factors),
                               bounds=(lower, upper))
        
        fitra, fitdec = self.orbitTI(res_fit.x, times)
        
        data = self.SimGaia(times, fdata, fitra, fitdec)
        # self.DataIntep = data
        return data 
    
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
            sim_astrometry.to_csv(data_dir+f'sim{self.ObjectName}_DR{str(self.DataRelease)}.dat', 
                                  sep=' ', header=False, index=False)
        return sim_astrometry
    
    def Plot(self, plot_dir=None, Npoints=500, scan_axis=None, scan_length=1):
        
        Period = self.ObjectParameters['P']
        
        timesOrb = np.linspace(-Period/2, Period/2, Npoints)
        dataOrb = self.SimPlot(timesOrb)
        
        timesSky = np.linspace(np.min(self.reltimes), np.max(self.reltimes), Npoints)
        dataSky = self.SimPlot(timesSky)
        
        if self.ObjectType=='cepheid':
            label1 = 'Cepheid'
            label2 = 'Companion'
            lw = 1
        elif self.ObjectType=='binary':
            label1 = 'Star 1'
            label2 = 'Star 2'
            lw = 1
        elif self.ObjectType=='BH':
            label1 = 'Star'
            label2 = 'Black hole'
            lw = 5
        elif self.ObjectType=='exoplanet':
            label1 = 'Host Star'
            label2 = 'Exoplanet'
            lw = 5
        else:
            raise KeyError(f"Unknown type {self.ObjectType}")
            
        fig, axs = plt.subplots(1,2, figsize=(14, 7), constrained_layout=True)
        fig.suptitle(self.ObjectName, fontsize=16)
        
        ax1, ax2 = axs
        
        ax1.set_title('Orbit motion')
        ax1.plot(dataOrb['ra1'], dataOrb['dec1'], label=label1, color = 'pink', lw = lw, zorder=1)
        ax1.plot(dataOrb['ra2'], dataOrb['dec2'], label=label2, color = 'lightskyblue', zorder=2)
        ax1.plot(dataOrb['ra_ph'], dataOrb['dec_ph'], label='Photocentre', color = 'black', zorder=3)
        ax1.scatter(self.Data['ra1'], self.Data['dec1'], color = 'pink', zorder=1, s=5)
        ax1.scatter(self.Data['ra2'], self.Data['dec2'], color = 'lightskyblue', zorder=2, s=5)
        ax1.scatter(self.Data['ra_ph'], self.Data['dec_ph'], color = 'black', zorder=3, s=10)
        ax1.xaxis.set_inverted(True)
        ax1.set_xlabel(r'$\Delta \alpha cos(\delta)$ [mas]')
        ax1.set_ylabel(r'$\Delta \delta$ [mas]')
        ax1.legend()
        ax1.set_aspect('equal', adjustable='datalim')
        
        
        ra_shift, dec_shift = np.mean(dataSky['ra_bs_plx']), np.mean(dataSky['dec_bs_plx'])
        
        ax2.set_title('On sky (orbit + proper + parallax motions)')
        ax2.plot(dataSky['ra_ss_plx']+ra_shift, dataSky['dec_ss_plx']+dec_shift, 
                    label='Single star model', color = 'plum', zorder=1)
        ax2.plot(dataSky['ra_bs_plx'], dataSky['dec_bs_plx'], 
                    label='Photocentre of the system', color = 'black', zorder=2)
        
        ax2.scatter(self.Data['ra_ss_plx']+ra_shift, self.Data['dec_ss_plx']+dec_shift, 
                    color = 'plum', zorder=1, s=5)
        ax2.scatter(self.Data['ra_bs_plx'], self.Data['dec_bs_plx'], 
                    color = 'black', zorder=2, s=5)
        ax2.xaxis.set_inverted(True)
        ax2.set_xlabel(r'$\Delta \alpha cos(\delta)$ [mas]')
        ax2.set_ylabel(r'$\Delta \delta$ [mas]')
        ax2.legend()
        ax2.set_aspect('equal', adjustable='datalim')
        
        
        if scan_axis is not None or scan_length !=1:
            if scan_axis is None or scan_axis=='all' or scan_axis==[1,2]:
                scan_axis=[True, True]
            elif scan_axis==1:
                scan_axis=[True, False]
            elif scan_axis==2:
                scan_axis=[False, True]
            else:
                raise ValueError(f"scan_axis can be 1, 2, [1,2], or 'all', currently {scan_axis}")
                
            if isinstance(scan_length, list):
                if len(scan_length) == 1:
                    scan_length=[scan_length[0], scan_length[0]]
                elif len(scan_length) != 2:
                    raise ValueError("scan_length can not contain more than 2 values.")
            else:
                scan_length=[scan_length, scan_length]
            scan_length = np.array(scan_length)
            
                
            ra = np.array([self.Data['ra_ph'], self.Data['ra_bs_plx']])
            dec = np.array([self.Data['dec_ph'], self.Data['dec_bs_plx']])
            
            for axi, ri, di, li in zip(axs[scan_axis], ra[scan_axis], dec[scan_axis], scan_length[scan_axis]):
                for r1, d1, ai in zip(ri, di, self.scanAngleRAD):
                    dx = 0.5 * li * np.sin(ai)
                    dy = 0.5 * li * np.cos(ai)
                    axi.plot([r1 - dx, r1 + dx], [d1 - dy, d1 + dy], color='gray', lw=1, alpha=0.5)
            
        if plot_dir is not None:
            fig.savefig(plot_dir+f'astrometry_gaia_{self.ObjectName}_DR{str(self.DataRelease)}.png', 
                        dpi=300, bbox_inches="tight")
        
    def PlotCepheid(self, plot_dir= None, Npoints=500):
        if not self.has_pulsation:
            raise KeyError(f"This plot is only for VIM (cepheid or mira). The current type is{self.ObjectType}")
        label1 = 'Cepheid'
        label2 = 'Companion'
        fig = plt.figure(constrained_layout=True, figsize=(14, 7))
        fig.suptitle(self.ObjectName, fontsize=16)
        gs = GridSpec(2, 2, figure=fig, width_ratios=[2, 3])
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[:, 1])
        
        # Pulsation
        Ppuls = self.ObjectParameters['Ppuls']
        
        timesPuls = np.linspace(-Ppuls/2, Ppuls/2, 1000)
        dataPuls = self.FluxRatio(timesPuls)
        
        Tplot = timesPuls[np.where(dataPuls['puls'] == np.min(dataPuls['puls']))]
        
        timesPuls = np.linspace(-Ppuls, Ppuls, Npoints)
        dataPuls = self.FluxRatio(timesPuls, Tplot = Tplot)

        ax1.set_title('Pulsation')
        ax1.plot(timesPuls, dataPuls['puls'], color = 'pink', lw = 3)
        ax1.set_xlabel('Time [day]')
        ax1.set_ylabel('Gmag [mag]')
        ax1.yaxis.set_inverted(True)
        
        # Orbit
        
        Period = self.ObjectParameters['P']
        
        timesOrb = np.linspace(-Period/2, Period/2, Npoints)
        dataOrb = self.SimPlot(timesOrb)
        
        temp = self.params_ph
        temp['a'] = self.a_ph_min
        ra_min, dec_min = self.orbit(temp, timesOrb)
        temp['a'] = self.a_ph_max
        ra_max, dec_max = self.orbit(temp, timesOrb)
        x_poly = np.concatenate([ra_min, ra_max[::-1]])
        y_poly = np.concatenate([dec_min, dec_max[::-1]])
        
        ax2.set_title('Orbit motion')
        ax2.fill(x_poly, y_poly, alpha=0.2, color = 'black', label = 'VIM zone', lw=0, zorder=2.5)
        ax2.plot(dataOrb['ra1'], dataOrb['dec1'], label=label1, color = 'pink', zorder=1)
        ax2.plot(dataOrb['ra2'], dataOrb['dec2'], label=label2, color = 'lightskyblue', zorder=2)
        ax2.plot(dataOrb['ra_ph_nps'], dataOrb['dec_ph_nps'], label='Mean photocentre', color = 'black', zorder=3)
        
        ax2.scatter(self.Data['ra1'], self.Data['dec1'], color = 'pink', zorder=1, s=5)
        ax2.scatter(self.Data['ra2'], self.Data['dec2'], color = 'lightskyblue', zorder=2, s=5)
        ax2.scatter(self.Data['ra_ph'], self.Data['dec_ph'], color = 'black', zorder=3, s=5)
        
        ax2.xaxis.set_inverted(True)
        ax2.set_xlabel(r'$\Delta \alpha cos(\delta)$ [mas]')
        ax2.set_ylabel(r'$\Delta \delta$ [mas]')
        ax2.set_aspect('equal', adjustable='datalim')
        ax2.legend()
        
        # Sky
        
        timesSky = np.linspace(np.min(self.reltimes), np.max(self.reltimes), Npoints)
        dataSky = self.SimPlot(timesSky)
        
        ra_shift, dec_shift = np.mean(dataSky['ra_nps_plx']), np.mean(dataSky['dec_nps_plx'])
        
        ax3.set_title('On sky (orbit + proper + parallax motions)')
        ax3.plot(dataSky['ra_ss_plx']+ra_shift, dataSky['dec_ss_plx']+dec_shift, 
                    label='Single star model', color = 'plum', zorder=1)
        ax3.plot(dataSky['ra_nps_plx'], dataSky['dec_nps_plx'], 
                 label='Binary system', color='darkviolet', zorder=2, lw=2)
        # ax3.plot(dataSky['ra_bs_plx'], dataSky['dec_bs_plx'], 
                    # label='Photocentre of the system', color = 'black')
        
        ax3.scatter(self.Data['ra_ss_plx']+ra_shift, self.Data['dec_ss_plx']+dec_shift, 
                    color = 'plum', zorder=1, s=5)
        ax3.scatter(self.Data['ra_nps_plx'], self.Data['dec_nps_plx'], 
                 color='darkviolet', zorder=2, s=5)
        ax3.scatter(self.Data['ra_bs_plx'], self.Data['dec_bs_plx'], 
                    color = 'black', zorder=3, s=5, label = 'VIM')
        
        ax3.xaxis.set_inverted(True)
        ax3.set_xlabel(r'$\Delta \alpha cos(\delta)$ [mas]')
        ax3.set_ylabel(r'$\Delta \delta$ [mas]')
        ax3.set_aspect('equal', adjustable='datalim')
        ax3.legend()
        
        if plot_dir is not None:
            fig.savefig(plot_dir+f'astrometry_gaia_cepheid_{self.ObjectName}_DR{str(self.DataRelease)}.png', 
                        dpi=300, bbox_inches="tight")
    
    def fitSS(self, w_bs):
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
        w_fit = mA @ p_fit
        return p_fit, w_fit
    
    def PlotSSfit(self, plot_dir=None):
        
        factra = -self.prlFactorAL*np.sin(self.scanAngleRAD)+self.prlFactorAC*np.cos(self.scanAngleRAD)
        factdec = self.prlFactorAL*np.cos(self.scanAngleRAD)+self.prlFactorAC*np.sin(self.scanAngleRAD)
        
        p_fit, w_fit = self.fitSS(self.w_bs)
        print(p_fit)
        r0, pmr, d0, pmd, plx = p_fit
        ra = r0 + pmr*self.reltimes + plx*factra
        dec = d0 + pmd*self.reltimes + plx*factdec
        
        fig, axs = plt.subplots(2,2, figsize=(14, 7), constrained_layout=True,
                                height_ratios=[4, 1])
        ax1, ax2, ax3, ax4 = axs.ravel()
        fig.suptitle(self.ObjectName, fontsize=16)
        
        ax1.set_title('On sky (orbit + proper + parallax motions)')
        ax1.plot(self.Data['ra_bs_plx'], self.Data['dec_bs_plx'], label = 'Data points', marker='.', color = 'coral')
        ax1.plot(ra, dec, label = 'Fit', marker='.', color = 'black')
        ax1.set_xlabel(r'$\Delta \alpha cos(\delta)$ [mas]')
        ax1.set_ylabel(r'$\Delta \delta$ [mas]')
        ax1.xaxis.set_inverted(True)
        ax1.legend()
        
        ax3.set_title('Along scan positions')
        ax2.scatter(self.reltimes, self.w_bs, label = 'Data points', color = 'coral', s=50)
        ax2.scatter(self.reltimes, w_fit, label = 'Fit', color = 'black', s=20)
        ax2.set_ylabel('AL positions [mas]', fontsize=14)
        ax2.legend()
        
        res2d = np.sqrt((self.Data['ra_bs_plx']-ra)**2 + (self.Data['dec_bs_plx']-dec)**2)
        ax3.scatter(self.reltimes, res2d, s=10, color = 'black')
        ax3.set_ylabel('Sky residuals [mas]', fontsize=12)

        res1d = self.w_bs-w_fit
        ax4.scatter(self.reltimes, res1d, s=10, color = 'black')
        ax4.set_ylabel('AL residuals [mas]', fontsize=12)
        
        if plot_dir is not None:
            fig.savefig(plot_dir+f'fitSS_{self.ObjectName}_DR{str(self.DataRelease)}.png', 
                        dpi=300, bbox_inches="tight")