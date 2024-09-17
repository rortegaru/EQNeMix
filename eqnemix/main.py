# Main module for eqnemix
# eqnemix/main.py
import numpy as np
import matplotlib.pyplot as plt
from nllgrid import NLLGrid
import geopandas as gpd
import pyekfmm as fmm
import pandas as pd
import json
import pyproj
import pymc

import pyekfmm as fmm
import pyproj



class eqnegrid:
   def __init__(self, fileextent='extent.shp', inputsrc=4326, outputcrs=3587, 
                  filesp='Vsp.npy',files='vs.npy', filep='vp.npy', deltadist=1000, **kwargs):
        """
        Constructor to initialize the EQNEGRID class.

        :param fileextent: Geographical Extension file, default is 'extent.shp'
        :param inputsrc: Input source, default is 4326
        :param outputcrs: Output coordinate reference system, default is 3587
        :param filesp: Numpy file, default is 'Vsp.npy'
        :param filep: P file, default is ''vp.npy'
        :param files: S file, default is 'vs.npy'
        :param deltadist: Number of cells in the grid, is the value in meters, default is 1000
        :param kwargs: Additional optional parameters.
        """
        self.fileextent = fileextent
        self.inputsrc = inputsrc
        self.outputcrs = outputcrs
        self.filesp = filesp
        self.filep = filep
        self.files = files
        self.deltadist = deltadist
        gdf = gpd.read_file(fileextent)
        self.inputcrs= gdf.crs
        gdf = gdf.to_crs(epsg=self.outputcrs)
        polygon = gdf.geometry.iloc[0]
        self.minx, self.miny, self.maxx, self.maxy = polygon.bounds
        #print(f"minx: {minx}, miny: {miny}")
        self.length_x = self.maxx - self.minx
        self.length_y = self.maxy - self.miny
        self.elements_x = int(self.length_x // self.deltadist)
        self.elements_y = int(self.length_y // self.deltadist)
        self.max_elements = max(self.elements_x, self.elements_y)
        #print(f"Elements in x: {elements_x}, Elements in y: {elements_y}")
        self.nx = 255
        self.ny = 255
        self.nz = 255
        
        self.dx, self.dy, self.dz = self.deltadist, self.deltadist,self.deltadist  
        self.x_orig=self.minx
        self.y_orig=self.miny
        self.z_orig=0
        self.gridp = NLLGrid(
        nx=self.nx, ny=self.ny, nz=self.nz,
        dx= self.dx, dy=self.dy, dz=self.dz,
        x_orig=self.x_orig, y_orig=self.y_orig, z_orig=self.z_orig

        )
        self.gridp.orig_lat = 0.0  # Asigna un valor predeterminado adecuado
        self.gridp.orig_lon = 0.0  # Asigna un valor predeterminado adecuado
        self.gridp.map_rot = 0.0   # Asigna un valor predeterminado adecuado
        self.gridp.proj_name = 'SIMPLE'  #

        self.gridp.init_array() 
        self.grids = NLLGrid(
        nx=self.nx, ny=self.ny, nz=self.nz,
        dx=self.dx, dy=self.dy, dz=self.dz,
        x_orig=self.x_orig, y_orig=self.y_orig, z_orig=self.z_orig
        )
        self.grids.orig_lat = 0.0  # Asigna un valor predeterminado adecuado
        self.grids.orig_lon = 0.0  # Asigna un valor predeterminado adecuado
        self.grids.map_rot = 0.0   # Asigna un valor predeterminado adecuado
        self.grids.proj_name = 'SIMPLE'  # Aseg??rate de que esta proyecci??n est?? definida y sea v??lida
        self.grids.init_array()
        self.gridp.float_type = 'FLOAT'  
        self.gridp.type = 'VELOCITY'  
        self.gridp.proj_name = 'SIMPLE'
        self.gridp.basename = 'Vp_grid'
        self.grids.float_type = 'FLOAT'  
        self.grids.type = 'VELOCITY'  
        self.grids.proj_name = 'SIMPLE'
        self.grids.basename = 'Vs_grid'
        self.grids.map_rot=0
        self.gridp.map_rot=0
        self.new_velp = self.gridp.array
        self.new_vels = self.grids.array
        np.save('vp.npy', self.new_velp)
        np.save('vs.npy', self.new_vels)
     #   print(type(new_velp))  
        print(self.new_velp.shape)  

   def print_variables(self):
        """
        Print Variables
        """
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")

   def print(*args, **kwargs):
        print_variables("EQNEMIX PARAMETERS:", *args, **kwargs)
   def savefiles(self,basefile):
        """
        Save al the files for next step

       :param basefile: 'Prefix for hdr and buf'

        
        """
        self.gridp.basename = basefile+'vp'
        self.grids.basename = basefile+'vs'
        self.gridp.write_hdr_file()
        self.gridp.write_buf_file()
        self.grids.write_hdr_file()
        self.grids.write_buf_file()
        self.new_velp = self.gridp.array
        self.new_vels = self.grids.array
        np.save(self.filep, self.new_velp)
        np.save(self.files, self.new_vels)

        variables_a_excluir = ['gridp', 'grids', 'new_velp', 'new_vels', 'inputcrs']

        # Crear un diccionario excluyendo las variables especificadas
        datos_a_guardar = {
            k: v for k, v in self.__dict__.items() if k not in variables_a_excluir
        }

        # Guarda las variables filtradas en un archivo JSON
        with open("varsgrid.json", 'w') as file:
            json.dump(datos_a_guardar, file)
 

class eqnefmm:
    def __init__(self, vpfile='vp.npy',vsfile='vs.npy',stax, stay):
        """
        Constructor to initialize the eqnefmm

        :param eqnegrid: Instance of grid previously created

        """
        #reshape and convert
        velp = np.load(vpfile)
        vels = np.load(vsfile)
        self.vpp = velp.reshape([255*255*255,1], order='F').astype('float32')
        self.vss = vels.reshape([255*255*255,1],order='F').astype('float32')

        #decidiendo si se utilizan valores de otra clase o si duplico variables



     # Lee los datos desde un archivo JSON
        with open('varsgrid.json', 'r') as file:
            datos = json.load(file)


        # Asigna cada par clave-valor del diccionario como un atributo de la clase
        for key, value in datos.items():
            setattr(self, key, value)  # Establece el atributo en el objeto actual

        print("Loaded variables:")
        for key in datos.keys():
            print(f"{key} = {getattr(self, key)}")  # Confirma que las variables est??n asignadas

# 
    def mostrar_atributos(self):
        # Listar los atributos actuales de la clase
        print("Atributos actuales de la clase:")
        atributos = vars(self)
        if not atributos:
            print("No se encontraron atributos asignados.")
        else:
            for attr, value in atributos.items():
                print(f"{attr} = {value}")  # Muestra cada atributo y su valor

      

    def _getminxy(self):
        gdf = gpd.read_file(self.eqnegrid.fileextent)
        gdf = gdf.to_crs(epsg=self.eqnegrid.outputcrs)
        polygon = gdf.geometry.iloc[0]
        minx, miny, maxx, maxy = polygon.bounds
        return minx, miny

    def _lonlat_to_xy(self,lon, lat, orig):
        minx = orig['minx']
        miny = orig['miny']
        stax = lon  
        stay = lat  
        input_crs = pyproj.CRS("EPSG:4326")
        output_crs = pyproj.CRS("EPSG:3857")
        transformer = pyproj.Transformer.from_crs(input_crs, output_crs, always_xy=True)
        stax_pro, stay_pro = transformer.transform(lon,lat)
        x_adj = stax_pro  - minx
        y_adj = stay_pro  - miny
        return x_adj,y_adj

    def _xy_to_lonlat(self,x, y, orig):
        # Extraer los valores de minx y miny desde el diccionario orig
        minx = orig['minx']
        miny = orig['miny']
        x_pro = x + minx
        y_pro = y + miny
        input_crs = pyproj.CRS("EPSG:3857")  
        output_crs = pyproj.CRS("EPSG:4326")  
        transformer = pyproj.Transformer.from_crs(input_crs, output_crs, always_xy=True)
        lon, lat = transformer.transform(x_pro, y_pro)
        return lon, lat

    def _calc_sp(self,vpp,vss,stay_pro,stax_pro,ngrid):
        print(vpp,vss,stay_pro,stax_pro,ngrid)
        # Caulculate P and S wave times based on Eikonal's function
 #       tp = fmm.eikonal(vpp, xyz=np.array([stay_pro,stax_pro,0]),ax=[0,1,ngrid],
#                         ay=[0,1,ngrid],az=[0,1,ngrid],order=2);
 #       ts = fmm.eikonal(vss, xyz=np.array([stay_pro,stax_pro,0]),ax=[0,1,ngrid],
 #                        ay=[0,1,ngrid],az=[0,1,ngrid],order=2);
 #       ttp = tp.reshape(ngrid,ngrid,ngrid,order='F')
 #       tts = ts.reshape(ngrid,ngrid,ngrid,order='F')
 #       tsp = tts - ttp
        return 0


    def save_all(self, basefile):
        """
        Example method to save all the pending files.
        
        :param basefile: Prefix for saving files.
        """
        # Llama al m??todo savefiles de eqnegrid con un prefijo personalizado
        self.eqnegrid.savefiles(basefile)
