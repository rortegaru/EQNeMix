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
        self.nx = self.max_elements
        self.ny = self.max_elements
        self.nz = self.max_elements
        
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
    def __init__(self,  stax, stay, velp='vp.npy',vels='vs.npy',fileextent='extent.shp', inputsrc=4326, outputcrs=3587,deltadist=1000):
        """
        Constructor to initialize the eqnefmm

        :param eqnegrid: Instance of grid previously created
        velp numpy array with P velocities
        vels numpy array with S velocities
        stax longitud of station, expected in EPSG:4326
        stay latitud of station, expected in EPSG:4326

        """
        self.deltadist = deltadist
        
        #Dimensiones del poligono
        gdf = gpd.read_file(fileextent)
        gdf = gdf.to_crs(epsg=self.outputcrs)
        polygon = gdf.geometry.iloc[0]
        self.minx, self.miny, self.maxx, self.maxy = polygon.bounds

        self.length_x = self.maxx - self.minx
        self.length_y = self.maxy - self.miny
        self.elements_x = int(self.length_x // self.deltadist)
        self.elements_y = int(self.length_y // self.deltadist)
        self.max_elements = max(self.elements_x, self.elements_y)
        #print(f"Elements in x: {elements_x}, Elements in y: {elements_y}")
        self.nx = self.max_elements
        self.ny = self.max_elements
        self.nz = self.max_elements

        #reshape and convert
        self.vpp = velp.reshape([self.nx*self.ny*self.nz,1], order='F').astype('float32')
        self.vss = vels.reshape([self.nx*self.ny*self.nz,1],order='F').astype('float32')


        self.inputsrc = pyproj.CRS(f"EPSG:{inputsrc}")
        self.outputcrs = pyproj.CRS(f"EPSG:{outputcrs}")

        transformer = pyproj.Transformer.from_crs(self.inputsrc, self.outputcrs, always_xy=True)
        # Transform the latitude and longitude station (CI.CLC) coodinates
        stax_pro, stay_pro = transformer.transform(stax, stay)

        stax_pro = stax_pro[0] - self.minx
        stay_pro = stay_pro[0] - self.miny
        self.stax_pro = int(stax_pro/1000)
        self.stay_pro = int(stay_pro/1000)

        #realizar el fmm
        self.tp = fmm.eikonal(self.vpp, xyz=np.array([self.stay_pro,self.stax_pro,0]),ax=[0,1,self.nx],ay=[0,1,self.ny],az=[0,1,self.nz],order=2);
        self.ts = fmm.eikonal(self.vss, xyz=np.array([self.stay_pro, self.stax_pro,0]),ax=[0,1,self.nx],ay=[0,1,self.ny],az=[0,1,self.nz],order=2);

        ttp = self.tp.reshape(255,255,255,order='F')
        tts = self.ts.reshape(255,255,255,order='F')
        self.tsp = tts - ttp


        


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

    def countour_graph(self):

        tn = self.tsp[:,:,1]
        tn = tn.reshape(self.nx, self.ny)

        # Plot contour times plot
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111,aspect=1.0)
        plt.contour(tn, levels=np.linspace(tn.min(), tn.max(), 10))
        plt.scatter(self.stax_pro,self.stay_pro,s=100,marker='^',color='black', label='CI.CLC')
        plt.xlabel('Delta index in X (longitude)')
        plt.ylabel('Delta index in Y (latitude)')
        plt.title('Contour plot of tn')

        # Show the plot
        plt.show()


    def full_graph(self):
        #Modified from Yangkang Chen, 2022. The University of Texas at Austin.

        # Define matrix dimensions
        Nx, Ny, Nz = self.nx, self.ny, self.nz
        X, Y, Z = np.meshgrid(np.arange(Nx)*.5, np.arange(Ny)*.5, np.arange(Nz)*.5)

        # Specify the 3D data
        data = self.tsp
        kw = {
            'vmin': data.min(),
            'vmax': data.max(),
            'alpha': 0.9,
            'levels': np.linspace(data.min(), data.max(), 20),
        }

        # Create a figure with 3D axes
        fig = plt.figure(figsize=(18, 9))
        # plt.subplot(1,2,1)
        plt.plasma()
        ax = fig.add_subplot(121, projection='3d')

        # Plot contour surfaces
        _ = ax.contourf(X[:, :, -1], Y[:, :, -1], data[:, :, 0], zdir='z', offset=0, **kw)
        _ = ax.contourf(X[0, :, :], data[0, :, :], Z[0, :, :], zdir='y', offset=0, **kw)
        C = ax.contourf(data[:, -1, :], Y[:, -1, :], Z[:, -1, :], zdir='x', offset=X.max(), **kw)

        # Set limits of the plot from coordinates limits
        xmin, xmax = X.min(), X.max()
        ymin, ymax = Y.min(), Y.max()
        zmin, zmax = Z.min(), Z.max()
        ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

        #Plot edges
        edges_kw = dict(color='0.4', linewidth=1, zorder=1e3)
        ax.plot([xmax, xmax], [ymin, ymax], 0, **edges_kw)
        ax.plot([xmin, xmax], [ymin, ymin], 0, **edges_kw)
        ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)

        # Set labels and zticks
        #ax.set(xlabel='X: Longitude [km]', ylabel='Y: Latitude [km]', zlabel='Z: Depth [km]')
        ax.set_xlabel('X: Longitude [km]')
        ax.set_ylabel('Y: Latitude [km]')
        ax.set_zlabel('Z: Depth [km]')

        # Set zoom and angle view
        # ax.view_init(40, -30, 0)
        # ax.set_box_aspect(None, zoom=0.9)         stax_pro,stay_pro

        #plt.gca().scatter(stax_pro, stay_pro, 0, s=100, marker='^', color='black', label='CI.CLC') #(x, y, z)
        plt.gca().set_xlim(0,131);
        plt.gca().set_ylim(0,131);
        plt.gca().set_zlim(0,131)
        plt.title('3D Travel Time S-P', color='k', fontsize=20)
        plt.gca().invert_zaxis()

        # Position for the colorbar
        #cb = plt.colorbar(C, cax = fig.add_axes([0.15,0.1,0.3,0.02]), format= "%4.2f", orientation='horizontal',label='Traveltime S-P [s]')
        cb = plt.colorbar(C, cax=fig.add_axes([0.10, 0.09, 0.25, 0.02]), format="%4.2f", orientation='horizontal', label='Traveltime S-P [s]')
        cb.ax.tick_params(labelrotation=45)
        # Save image plot
        plt.savefig('test_pyekfmm_fig2.png',format='png',dpi=300,bbox_inches='tight', pad_inches=0.3)
        plt.savefig('test_pyekfmm_fig2.pdf', format='pdf', dpi=350, bbox_inches='tight', pad_inches=0.1)


        # Show Figure
        plt.show()



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


class eqnemaster:
    def __init__(self,station_lon, station_lat, EPSG=4326 ,deltadist=1000):
        """
        :param station_lon: Longitud of the station to work with
        :param station_lat: Latitud of the station to work with
        :param deltadist: Number of cells in the grid, is the value in meters, default is 1000
        """
        self.stalon = station_lon
        self.stalat = station_lat
        self.stacrs = EPSG
        self.deltadist = deltadist



    def xminyminshape(self,file,inputcrs=4326,outputcrs=3857):
        """
        :param fileextent: Geographical Extension file, like 'extent.shp'        
        :param inputsrc: Input source, default is 4326
        :param outputcrs: Output coordinate reference system, default is 3857
        """
        
        self.inputcrsshape = inputcrs
        self.outputcrsshape = outputcrs

        gdf = gpd.read_file(file) #Read the file
        gdf = gdf.set_crs(f"EPSG:{self.inputcrsshape}")

        # Verify the projection and change it if necessary
        if gdf.crs.is_geographic:
            gdf = gdf.to_crs(epsg=self.outputcrsshape )
        else:
            print("The shapefile is already projected")

        # Get the coordinates of the bounds of the polygon
        polygon = gdf.geometry.iloc[0]
        minx, miny, maxx, maxy = polygon.bounds

        # Safe the coordiantes of the bottom left cornder
        self.minx = minx
        self.miny = miny
        # Get the horizontal and vertical length fo the area
        length_x = maxx - minx
        length_y = maxy - miny

        # Calculate the number of elements along the x and y axis
        elements_x = int(length_x // self.deltadist)
        elements_y = int(length_y // self.deltadist)

        # Define the working dimentions (keeping in mind that we need to work with a cube)
        self.max_elements = max(elements_x, elements_y)
        self.nx = self.max_elements
        self.ny = self.max_elements
        self.nz = self.max_elements

        return self.minx, self.miny, self.max_elements
    
    def xminymin(self,xmin,ymin,xmax,ymax,inputcrs=4326,outputcrs=3857):
        self.inputcrsshape = inputcrs
        self.outputcrsshape = outputcrs
        # Define the input coordinates system (latitude and longitude)
        input_crs = pyproj.CRS(f"EPSG:{self.inputcrsshape}")  # EPSG:4326 represents WGS 84 (latitude and longitude)
        # Define the output coordinates system (latitude and longitude)
        output_crs = pyproj.CRS(f"EPSG:{self.outputcrsshape}")
        # Create a coordinates transformer
        transformer = pyproj.Transformer.from_crs(input_crs, output_crs, always_xy=True, accuracy=1e-9)

        # Transform the latitude and longitude coodinates
        self.minx, self.miny = transformer.transform(xmin, ymin)

        self.maxx, self.maxy = transformer.transform(xmax, ymax)

        # Calculate the size of the rectangle in meters
        length_x = self.maxx - self.minx
        length_y = self.maxy - self.miny

        # Calculate the number of elements along the x and y axis
        elements_x = int(length_x // self.deltadist)
        elements_y = int(length_y // self.deltadist)

        # Define the working dimentions (keeping in mind that we need to work with a cube)
        self.max_elements = max(elements_x, elements_y)
        self.nx = self.max_elements
        self.ny = self.max_elements
        self.nz = self.max_elements

        return self.minx, self.miny, self.max_elements
    

    def nllgrid(self,vp_values,vs_values,layer_thickness):
        if len(layer_thickness) != len(vp_values):
            raise ValueError("The size of layer_thickness must be the same as the size of vp_values")
        if len(layer_thickness) != len(vs_values):
            raise ValueError("The size of layer_thickness must be the same as the size of vs_values")

        # P velocity array  
        gridp = NLLGrid(
            nx=self.nx, ny=self.ny, nz=self.nz,
            dx=self.deltadist, dy=self.deltadist, dz=self.deltadist,
            x_orig=0, y_orig=0, z_orig=0
        )
        gridp.init_array() #Inicialize the tridimentional array with zeros
       
        z_start = 0
        # Assign the velocities to the corresponding position in the array
        for i, vs in enumerate(vp_values):
            z_end = z_start + layer_thickness[i]
            gridp.array[:, :, z_start:z_end] = vs
            z_start = z_end  # Change the z_atart for the next layer
        if z_end < self.nz:
            gridp.array[:, :, z_end:] = vp_values[-1]

        # S velocity array
        grids = NLLGrid(
            nx=self.nx, ny=self.ny, nz=self.nz,
            dx=self.deltadist, dy=self.deltadist, dz=self.deltadist,
            x_orig=0, y_orig=0, z_orig=0
        )  
        grids.init_array()  # Inicialize the tridimentional array with zeros

        # Assign the velocities to the corresponding position in the array
        z_start = 0
        for i, vs in enumerate(vs_values):
            z_end = z_start + layer_thickness[i]
            grids.array[:, :, z_start:z_end] = vs
            z_start = z_end  # Change the z_atart for the next layer
        if z_end < self.nz:
            grids.array[:, :, z_end:] = vs_values[-1]

        self.gridp = gridp.array
        self.grids = grids.array
        
        return gridp, grids
    
    def eqnefmm(self):

        #reshape arrays

        self.vpp = self.gridp.reshape([255*255*255,1], order='F').astype('float32')
        self.vss = self.grids.reshape([255*255*255,1], order='F').astype('float32')

        #change the station coordinate points

        input_crs = pyproj.CRS(f"EPSG:{self.stacrs}")  # EPSG:4326 represents WGS 84 (latitude and longitude)

        # Define the output coordinates system (latitude and longitude)
        output_crs = pyproj.CRS(f"EPSG:{self.outputcrsshape}")
        # Create a coordinates transformer
        transformer = pyproj.Transformer.from_crs(input_crs, output_crs, always_xy=True)

        # Transform the latitude and longitude station (CI.CLC) coodinates
        stax_pro, stay_pro = transformer.transform(self.stalon, self.stalat)


        stax_pro = stax_pro - self.minx
        stay_pro = stay_pro - self.miny
        # Switch metrers for kilometros
        self.stax_pro = int(stax_pro/1000) 
        self.stay_pro = int(stay_pro/1000)

        # Carry out the fast marching
        self.tp = fmm.eikonal(self.vpp, xyz=np.array([self.stay_pro,self.stax_pro,0]),ax=[0,1,self.nx],ay=[0,1,self.ny],az=[0,1,self.nz],order=2);
        self.ts = fmm.eikonal(self.vss, xyz=np.array([self.stay_pro, self.stax_pro,0]),ax=[0,1,self.nx],ay=[0,1,self.ny],az=[0,1,self.nz],order=2);

        ttp = self.tp.reshape(self.nx,self.ny,self.nz,order='F')
        tts = self.ts.reshape(self.nx,self.ny,self.nz,order='F')
        self.tsp = tts - ttp
        
        return self.tsp
    
