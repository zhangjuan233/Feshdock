"""Module to package a protein complex"""
import numpy as np
from feshdock.structure.space import SpacePoints
from docking.outputs.constant import ParamterStore
from scipy.spatial import distance
import math,threading
import pyfftw
from concurrent.futures import ThreadPoolExecutor

class Complex(object):
    """Represents a molecular complex"""

    def __init__(
        self,
        chains,
        atoms=None,
        residues=None,
        structure_file_name="",
        structures=None,
        representative_id=0,
        atomtype_list=None
    ):
        """Creates a new complex that can deal with multiple coordinates for a given atom"""
        self.chains = chains
        # Set atoms at the upper level for fast indexing
        if atoms:
            self.atoms = atoms
        else:
            self.atoms = [
                atom
                for chain in self.chains
                for residue in chain.residues
                for atom in residue.atoms
            ]
        for atom_index, atom in enumerate(self.atoms):
            atom.index = atom_index

        # Same for residues
        if residues:
            self.residues = residues
        else:
            self.residues = [
                residue for chain in self.chains for residue in chain.residues
            ]
        for residue_index, residue in enumerate(self.residues):
            residue.index = residue_index

        if structures:
            self.num_structures = len(structures)
            self.structure_file_names = [
                structure["file_name"] for structure in structures
            ]
            self.atom_coordinates = [
                SpacePoints([[atom.x, atom.y, atom.z] for atom in structure["atoms"]])
                for structure in structures
            ]
        else:
            self.num_structures = 1
            self.structure_file_names = [structure_file_name]
            self.atom_coordinates = [
                SpacePoints([[atom.x, atom.y, atom.z] for atom in self.atoms])
            ]

        self.num_atoms = len(self.atoms)
        self.protein_num_atoms = sum(
            [len(residue.atoms) for residue in self.residues if residue.is_protein()]
        )
        self.nucleic_num_atoms = sum(
            [len(residue.atoms) for residue in self.residues if residue.is_nucleic()]
        )
        self.num_residues = len(self.residues)
        self.representative_id = representative_id
        self.nm_mask = self.get_nm_mask()
        # ///////////////////////////////
        self.atom_type = atomtype_list  # 添加一个残基原子类型
        self.grid_width=1.2#网格宽度
        self.radius=[]
        self.charge=[]
        self.ace=[]
        self.atomtype=[]#依据原子类型赋值（0,1,2）
        self.para=ParamterStore()
        self.num_grid=None
        self.grid_coord=[]
        self.grid_r=[]
        self.grid_i=[]
        self.fft_result=None
        self.flag=None
        self.lock=threading.Lock()

    @staticmethod
    def from_structures(structures, representative_id=0):
        return Complex(
            structures[representative_id]["chains"],
            structures[representative_id]["atoms"],
            structures[representative_id]["residues"],
            structures[representative_id]["file_name"],
            structures,
            representative_id,
            structures[representative_id]["atomtype_list"]
        )

    def clone(self):
        """Creates a copy of the current complex"""
        molecule = Complex([chain.clone() for chain in self.chains])
        molecule.num_structures = self.num_structures
        molecule.structure_file_names = self.structure_file_names
        molecule.atom_coordinates = self.copy_coordinates()
        return molecule

    def copy_coordinates(self):
        """Deep copy of atom coordinates"""
        return [coordinates.clone() for coordinates in self.atom_coordinates]

    def get_nm_mask(self):
        """Calculates the mask on atoms to apply ANM"""
        mask = []
        for residue in self.residues:
            # is_protein should accept common modified cases supported by ProDy
            if residue.is_protein() or residue.is_nucleic():
                mask.extend([True] * len(residue.atoms))
            else:
                mask.extend([False] * len(residue.atoms))
        return np.array(mask)

    def get_atoms(self, protein=True, nucleic=True, dummy=False):
        """Selects atoms on structure depending on their nature"""
        atoms = []
        for residue in self.residues:
            if residue.is_standard and protein:
                atoms.extend(residue.atoms)
            elif residue.is_nucleic and nucleic:
                atoms.extend(residue.atoms)
            elif residue.is_dummy and dummy:
                atoms.extend(residue.atoms)
        return atoms

    def center_of_mass(self, structure=None):
        """Calculates the center of mass"""
        if not structure:
            structure = self.representative_id
        if len(self.atoms):
            total_x = 0.0
            total_y = 0.0
            total_z = 0.0
            total_mass = 0.0
            for atom in self.atoms:
                total_x += self.atom_coordinates[structure][atom.index][0] * atom.mass
                total_y += self.atom_coordinates[structure][atom.index][1] * atom.mass
                total_z += self.atom_coordinates[structure][atom.index][2] * atom.mass
                total_mass += atom.mass
            return [total_x / total_mass, total_y / total_mass, total_z / total_mass]
        else:
            return [0.0, 0.0, 0.0]

    def center_of_coordinates(self, structure=None):
        """Calculates the center of coordinates"""
        if not structure:
            structure = self.representative_id
        atoms = [atom for atom in self.atoms if not atom.is_hydrogen()]
        dimension = len(atoms)
        if dimension:
            total_x = 0.0
            total_y = 0.0
            total_z = 0.0
            for atom in atoms:
                total_x += self.atom_coordinates[structure][atom.index][0]
                total_y += self.atom_coordinates[structure][atom.index][1]
                total_z += self.atom_coordinates[structure][atom.index][2]
            return [total_x / dimension, total_y / dimension, total_z / dimension]
        else:
            return [0.0, 0.0, 0.0]

    def translate(self, vector):
        """Translates atom coordinates based on vector"""
        for coordinates in self.atom_coordinates:
            coordinates.translate(vector)#沿着指定向量平移点坐标

    def rotate(self, q):
        """Rotates this complex using a quaternion q"""
        for coordinates in self.atom_coordinates:
            coordinates.rotate(q)

    def move_to_origin(self):
        """Moves the structure to the origin of coordinates"""
        translation = [-1 * c for c in self.center_of_coordinates()] # 计算质心  计算需要平移的向量（取反）
        self.translate(translation)
        return translation

    def get_residue(self, chain_id, residue_name, residue_number, residue_insertion=""):
        for chain in self.chains:
            if chain_id == chain.cid:
                for residue in chain.residues:
                    if (
                        residue.name == residue_name
                        and int(residue.number) == int(residue_number)
                        and residue.insertion == residue_insertion
                    ):
                        return residue
        return None

    def __getitem__(self, item):
        return self.atom_coordinates[item]

    def __setitem__(self, index, item):
        self.atom_coordinates[index] = item

    def __iter__(self):
        for coordinates in self.atom_coordinates:
            yield coordinates

    def __len__(self):
        return self.atom_coordinates.shape[0]

    def representative(self, is_membrane=False):
        coordinates = self.atom_coordinates[self.representative_id]
        if is_membrane:
            transmembrane = []
            for atom_id, atom in enumerate(self.atoms):
                if atom.residue_name != "MMB":
                    transmembrane.append(coordinates[atom_id])
            return transmembrane
        else:
            return coordinates

    # /////////////////////////
    # 根据受体的原子类型计算半径，电荷，以及ace, 以及为每种原子类型赋值3种（0,1,2）
    def mycopy_dist(self):
        for i in range(self.num_atoms):
            self.radius.append(self.para.radius_dist[self.atom_type[i]])
            self.charge.append(self.para.charge_dist[self.atom_type[i]])
            self.ace.append(self.para.ace_dist[self.atom_type[i]])
            if self.atoms[i].name == 'CA':  # 依据原子类型赋值，1:CA, 2:C,N, 0:protein
                self.atomtype.append(1)
            elif self.atoms[i].name == 'C' or self.atoms[i].name == 'N':
                self.atomtype.append(2)
            else:
                self.atomtype.append(0)

    # 计算配体所需的网格数量
    def autogrid_l(self):
        self.flag=0

        matrix_coord=self.representative()
        dis=distance.pdist(matrix_coord)
        dis_max=np.max(dis)#最远两原子距离
        dis_max+=2.0*3
        num_grid=1+int(dis_max/self.grid_width)
        for i in range(len(self.para.gridtable)):
            if self.para.gridtable[i]>=num_grid:
                num_grid=self.para.gridtable[i]
                break
        self.num_grid=num_grid

    def autogrid_r(self):
        # 计算边界值
        self.flag=1

        BIG=1e+9
        Edge=np.zeros((3,2))
        Edge[:,0]=BIG#上界
        Edge[:,1]=-(1e+9)
        for i in range(self.num_atoms):#x  y   z
            Edge[0][0]=min(self.atom_coordinates[0][i][0],Edge[0][0])
            Edge[1][0]=min(self.atom_coordinates[0][i][1],Edge[1][0])
            Edge[2][0]=min(self.atom_coordinates[0][i][2],Edge[2][0])
            # 下界
            Edge[0][1] = max(self.atom_coordinates[0][i][0], Edge[0][1])
            Edge[1][1] =max(self.atom_coordinates[0][i][1], Edge[1][1])
            Edge[2][1] =max(self.atom_coordinates[0][i][2], Edge[2][1])

        size_rec=0
        for i in range(3):
            size=Edge[i][1]-Edge[i][0]
            size_rec=max(size,size_rec)
        size_rec += 2.0 * 8
        num_grid = 1 + int(size_rec / self.grid_width)
        for i in range(len(self.para.gridtable)):
            if self.para.gridtable[i] >= num_grid:
                num_grid = self.para.gridtable[i]
                break
        self.num_grid = num_grid

        # matrix_coord=self.representative()
        # dis=distance.pdist(matrix_coord)
        # dis_max=np.max(dis)#最远两原子距离 受体55
        # dis_max+=2.0*8
        # num_grid=1+int(dis_max/self.grid_width)
        # for i in range(len(self.para.gridtable)):
        #     if self.para.gridtable[i]>=num_grid:
        #         num_grid=self.para.gridtable[i]
        #         break
        # self.num_grid=num_grid

    # 受体初始化
    def rec_init(self,num_grid):
        self.para.num_grid=num_grid
        self.para.num_fft=2*num_grid
        search_length=self.grid_width*num_grid #每维的搜索范围
        # 计算受体网格坐标
        Grid_coord=[]
        for i in range(num_grid):
            Grid_coord.append(search_length*(-0.5+(0.5+i)/num_grid))
        self.grid_coord=Grid_coord
        # 根据评分函数计算每个网格的值
        self.create_voxel()
        # 对受体进行FFT操作
        self.rec_fft()

    def rec_fft(self):
        ng3=self.para.num_grid**3
        # complex_array = np.array(self.grid_r) + 1j * np.array(self.grid_i)
        # output_array = np.empty_like(complex_array)
        # fft = pyfftw.builders.fftn(complex_array, planner_effort='FFTW_ESTIMATE')
        # fft(output_array)

        complex_array1=np.array(self.grid_r) + 1j * np.array(self.grid_i)
        aligned_array = pyfftw.empty_aligned(ng3, dtype='complex128')
        aligned_array[:] = complex_array1
        fft_result = pyfftw.interfaces.numpy_fft.fft(aligned_array)# 受体fft后的结果
        self.fft_result=fft_result

    def create_voxel(self):
        beta=-2800
        if self.flag==1:
            self.rpscace_r(self.para.ACE_ratio,self.para.rec_core,self.para.ligand_core)
            self.electro_r(beta,self.para.Elec_ratio)
        elif self.flag==0:
            self.rpscace_l(self.para.ACE_ratio, self.para.rec_core, self.para.ligand_core_l)
            self.electro_l(beta, self.para.Elec_ratio)



    def rpscace_r(self,ACE_ratio,rec_core,ligand_core):
        rho=-3.0
        epsilion=rec_core#-45
        open_space=-7777.0

        ng1=self.para.num_grid#网格数量
        ng2=ng1*ng1
        ng3=ng1*ng1*ng1
        na=self.num_atoms
        nag=na*ng1
        self.precalc_r()
        self.voxel_init()
        self.grid_r = [0 for _ in range(ng3)]
        # 将在原子内部的网格赋值-45
        for l in range(self.num_atoms):
            search_range=int((2.4+self.grid_width-0.01)/self.grid_width)
            # 计算原子坐标在网格中的索引
            i2=int(self.atom_coordinates[0][l][0]/self.grid_width+ng1/2)
            j2=int(self.atom_coordinates[0][l][1]/self.grid_width+ng1/2)
            k2=int(self.atom_coordinates[0][l][2]/self.grid_width+ng1/2)
            # 计算在网格搜索的上下界
            ia=max(i2-search_range,0)#下界
            ja=max(j2-search_range,0)#下界
            ka=max(k2-search_range,0)#下界
            ib=min(i2+search_range+1,ng1)#上界
            jb=min(j2+search_range+1,ng1)#上界
            kb=min(k2+search_range+1,ng1)#上界
            lc=ng1*l

            # grid_r grid_i存储三维网格数据  应该是ng3的大小吧   初始化
            # self.grid_r = [[[0 for _ in range(ng1)] for _ in range(ng1)] for _ in range(ng1)]
            # self.grid_r=np.array(self.grid_r)

            # 从三个维度上搜索
            radius_core2_l = self.para.radius_core2[l]
            xd_lc = self.para.xd[lc+ia:lc + ib ]
            yd_lc = self.para.yd[lc+ja:lc + jb]
            zd_lc = self.para.zd[lc+ka:lc + kb]
            # 优化后的循环
            for idx, i in enumerate(range(ia, ib)):
                if xd_lc[idx] > radius_core2_l:
                    continue
                for jdx,j in enumerate(range(ja,jb)):
                    d2 = xd_lc[idx] + yd_lc[jdx]
                    if d2 > radius_core2_l:
                        continue
                    for kdx,k in enumerate( range(ka, kb)):
                        d3 = d2 + zd_lc[kdx]
                        if d3 < radius_core2_l:
                            self.grid_r[ng2 * i + ng1 * j + k] = epsilion

            # for i in range(ia,ib):
            #     if self.para.xd[lc+i]>self.para.radius_core2[l]:#xd的大小是原子数*网格数
            #         continue
            #     for j in range(ja,jb):
            #         d2=self.para.xd[lc+i]+self.para.yd[lc+j]
            #         if d2>self.para.radius_core2[l]:
            #             continue
            #         for k in  range(ka,kb):
            #             d3=d2+self.para.zd[lc+k]
            #             if d3<self.para.radius_core2[l]:  #给当前的网格赋值-45
            #                 # self.grid_r[i][j][k]=epsilion
            #                 self.grid_r[ng2*i+ng1*j+k]=epsilion

        # num= 0
        # for i in range(ng3):
        #     if self.grid_r[i] == epsilion:
        #        num += 1

        # fill up cavity  经过这步操作网格为-45或-7777
        for ij in range(ng2):
            i =ij%ng1#取余数
            j=ij//ng1
            # x,forward
            for k in range(ng1):
                if self.grid_r[i+ng1*j+ng2*k]!=epsilion:
                    self.grid_r[i + ng1 * j + ng2 * k]=open_space
                else:
                    break
            for k in range(ng1-1,-1):#x,反方向
                if self.grid_r[i + ng1 * j + ng2 * k] != epsilion:
                    self.grid_r[i + ng1 * j + ng2 * k] = open_space
                else:
                    break


            # y方向
            for k in range(ng1):
                if self.grid_r[k+ng1*i+ng2*j]!=epsilion:
                    self.grid_r[k + ng1 * i + ng2 * j]=open_space
                else:
                    break
            for k in range(ng1 - 1, -1):
                if self.grid_r[k + ng1 * i + ng2 * j] != epsilion:
                    self.grid_r[k + ng1 * i + ng2 * j] = open_space
                else:
                    break
            # z方向
            for k in range(ng1):
                if self.grid_r[j+ng1*k+ng2*i]!=epsilion:
                    self.grid_r[j + ng1 * k + ng2 * i]=open_space
                else:
                    break
            for k in range(ng1 - 1, -1):
                if self.grid_r[j + ng1 * k + ng2 * i] != epsilion:
                    self.grid_r[j + ng1 * k + ng2 * i] = open_space
                else:
                    break
        # k=0
        # for i in range(ng3):
        #     if self.grid_r[i]==0:
        #         k=k+1

        # 根据"cavity"区域中已经标记为开放空间的网格，通过填充相邻网格来扩展开放空间的范围
        for ink in range(2):
            for ijk in range(ng2,ng3-ng2):
                j=(ijk / ng1) % ng1
                if j < 2 or j > ng1 - 2:continue
                k = ijk % ng1;
                if k < 2 or k > ng1 - 2 :continue
                if self.grid_r[ijk]==open_space:
                    if  self.grid_r[ijk+1]==0:
                        self.grid_r[ijk + 1]=open_space
                    if self.grid_r[ijk - 1] == 0:
                        self.grid_r[ijk -1] = open_space
                    if self.grid_r[ijk + ng1] == 0:
                        self.grid_r[ijk + ng1] = open_space
                    if self.grid_r[ijk - ng1] == 0:
                        self.grid_r[ijk - ng1] = open_space
                    if self.grid_r[ijk + ng2] == 0:
                        self.grid_r[ijk + ng2] = open_space
                    if self.grid_r[ijk - ng2] == 0:
                        self.grid_r[ijk -ng2] = open_space

        # 将值为0的变成-45
        for i in range(ng3):
            if self.grid_r[i]==0:
                self.grid_r[i]=epsilion
        for i in range(ng3):
            if self.grid_r[i]==open_space:
                self.grid_r[i]=0
        k=0
        for i in range(ng3):
            if self.grid_r[i]==epsilion:
                k=k+1

        # 识别在原子表面的网格点,计算rpsdc+ace的值
        voxel_rpscace=[0 for _ in range(ng3)]
        for l in range(self.num_atoms):
            search_range = int((12 + self.grid_width - 0.01) / self.grid_width)
            # 计算原子坐标在网格中的索引
            i2 = int(self.atom_coordinates[0][l][0] / self.grid_width + ng1 / 2)
            j2 = int(self.atom_coordinates[0][l][1] / self.grid_width + ng1 / 2)
            k2 = int(self.atom_coordinates[0][l][2] / self.grid_width + ng1 / 2)
            # 计算在网格搜索的上下界
            ia = max(i2 - search_range, 0)  # 下界
            ja = max(j2 - search_range, 0)  # 下界
            ka = max(k2 - search_range, 0)  # 下界
            ib = min(i2 + search_range + 1, ng1)  # 上界
            jb = min(j2 + search_range + 1, ng1)  # 上界
            kb = min(k2 + search_range + 1, ng1)  # 上界
            lc = ng1 * l

            # grid_r grid_i存储三维网格数据  应该是ng3的大小吧   初始化
            # self.grid_r = [[[0 for _ in range(ng1)] for _ in range(ng1)] for _ in range(ng1)]
            # self.grid_r = [0 for _ in range(ng3)]
            # self.grid_r=np.array(self.grid_r)

            # 从三个维度上搜索
            radius_judge_l = self.para.rjudge2[l]
            xd_lc = self.para.xd[lc + ia:lc + ib]
            yd_lc = self.para.yd[lc + ja:lc + jb]
            zd_lc = self.para.zd[lc + ka:lc + kb]
            # 优化后的循环
            for idx, i in enumerate(range(ia, ib)):
                if xd_lc[idx] >  radius_judge_l :
                    continue
                for jdx, j in enumerate(range(ja, jb)):
                    d2 = xd_lc[idx] + yd_lc[jdx]
                    if d2 >  radius_judge_l :
                        continue
                    for kdx, k in enumerate(range(ka, kb)):
                        d3 = d2 + zd_lc[kdx]
                        if d3 <  radius_judge_l :
                            voxel_rpscace[ng2 * i + ng1 * j + k] += 1.0+self.ace[l]*(-0.8)*ACE_ratio
                            # num_voxe+=1
        num_voex=0
        for i in range(ng3):
            # if voxel_rpscace[i]!=0:
            #     num_voex+=1
            if self.grid_r[i]==epsilion:
                continue
            self.grid_r[i]=voxel_rpscace[i]
        # for i in range(ng3):
        #     if self.grid_r[i]!=0:
        #         num_voex+=1#36291

    # 计算受体的静电分数
    def electro_r(self,beta,elec_ratio):
        ftr1 = 6.0
        ftr2 = 8.0
        er1 = 4.0
        er2 = 38.0
        er3 = -224.0
        er4 = 80.0
        EPS=9.99999997e-07

        search_range=0
        ng1=self.para.num_grid
        ng2=ng1*ng1
        ng3=ng2*ng1
        grid_width=self.grid_coord[1]-self.grid_coord[0]
        charge_per_grid = -1 * grid_width * er4 / (beta * elec_ratio)
        self.grid_i=[0 for _ in range(ng3)]
        for l in range(self.num_atoms):
            abs_charge=abs(self.charge[l])
            if abs_charge<=EPS:
                continue
            if abs_charge<0.07:
                search_range=4
            elif abs_charge<0.21:
                search_range=5
            else:
                search_range=int(abs_charge/charge_per_grid)

            # 遍历网格，为网格分配分数
            # 计算原子坐标在网格中的索引
            i2 = int(self.atom_coordinates[0][l][0] / self.grid_width + ng1 / 2)
            j2 = int(self.atom_coordinates[0][l][1] / self.grid_width + ng1 / 2)
            k2 = int(self.atom_coordinates[0][l][2] / self.grid_width + ng1 / 2)
            # 计算在网格搜索的上下界
            ia = max(i2 - search_range, 0)  # 下界
            ja = max(j2 - search_range, 0)  # 下界
            ka = max(k2 - search_range, 0)  # 下界
            ib = min(i2 + search_range + 1, ng1)  # 上界
            jb = min(j2 + search_range + 1, ng1)  # 上界
            kb = min(k2 + search_range + 1, ng1)  # 上界
            lc = ng1 * l

            # grid_r grid_i存储三维网格数据  应该是ng3的大小吧   初始化
            # self.grid_r = [[[0 for _ in range(ng1)] for _ in range(ng1)] for _ in range(ng1)]
            # self.grid_r=np.array(self.grid_r)

            # 从三个维度上搜索
            xd_lc = self.para.xd[lc + ia:lc + ib]
            yd_lc = self.para.yd[lc + ja:lc + jb]
            zd_lc = self.para.zd[lc + ka:lc + kb]
            # 优化后的循环
            for idx, i in enumerate(range(ia, ib)):
                for jdx, j in enumerate(range(ja, jb)):
                    d2 = xd_lc[idx] + yd_lc[jdx]
                    for kdx, k in enumerate(range(ka, kb)):
                        ijk=ng2*i+ng1*j+k
                        if self.grid_r[ijk]<0:
                            continue
                        d3 = d2 + zd_lc[kdx]
                        d3=math.sqrt(d3)
                        if d3>=ftr2:
                            er=er4
                        elif d3>ftr1:
                            er=er2*d3+er3
                        else:
                            er=er1
                        elec=elec_ratio*beta*self.charge[l]/er/d3
                        self.grid_i[ijk]+=elec
            # num=0#755
            # for i in range(ng3):
            #     if self.grid_i[i]!=0:
            #         num+=1


    def precalc_r(self):# 用来判断网格是否在原子的核心、表面和外部
        D=3.6
        rcore2=1.5
        rsurf2=0.8

        for i in range(self.num_atoms):
           self.para.rjudge2.append(( self.radius[i]+D)*( self.radius[i]+D))
           self.para.radius_core2.append(self.radius[i]*self.radius[i]*rcore2)
           self.para.radius_surf2.append(self.radius[i]*self.radius[i]*rsurf2)

    def voxel_init(self):#计算当前原子与每个网格在三个维度上的差值的平方
        nag=self.num_atoms*self.para.num_grid
        for i in range(self.num_atoms):
            for j in range(self.para.num_grid):
                self.para.xd.append(np.float64(self.atom_coordinates[0][i][0]-self.grid_coord[j]))#列表大小为原子数*网格数
                self.para.yd.append(self.atom_coordinates[0][i][1]-self.grid_coord[j])
                self.para.zd.append(self.atom_coordinates[0][i][2]-self.grid_coord[j])
        for i in range(nag):
            self.para.xd[i]=np.multiply(self.para.xd[i],self.para.xd[i])
            # self.para.xd[i]*=self.para.xd[i]
            self.para.yd[i]*=self.para.yd[i]
            self.para.zd[i]*=self.para.zd[i]

    # def dockz(self,num_grid):
    #     self.para.num_grid=num_grid
    #     nc=max(self.para.num_rot_angles//10,1)
    #     pool=Pool()
    #     result_queue=Queue()
    #     for ang in range(self.para.num_rot_angles):
    #         pool.apply_async(self.process_rotation, args=(ang, result_queue))
    #
    #         # 关闭进程池
    #     pool.close()
    #     pool.join()
    #
    #     # 从队列中获取结果
    #     results = []
    #     while not result_queue.empty():
    #         result = result_queue.get()
    #         results.append(result)
    #
    #     print('pro')

        # pool.map(self.process_rotation,range(self.para.num_rot_angles))
        # for ang in range(self.para.num_rot_angles):
        #     theta=ParamterStore.zangle[ang] #旋转角度
        #     self.ligand_rotation(theta) #旋转配体
        #     self.create_voxel()#计算得分
        #     # 配体进行FFT操作

    # def dockz(self, num_grid):
    #     self.para.num_grid = num_grid
    #     for ang in range(self.para.num_rot_angles):
    #         theta=ParamterStore.zangle[ang] #旋转角度
    #         self.ligand_rotation(theta) #旋转配体
    #         self.create_voxel()#计算得分
    #     #     # 配体进行FFT操作

    def dockz(self, num_grid):
        self.para.num_grid = num_grid
        ng1=self.para.num_grid
        ng2=ng1**2
        Grid_coord = []
        search_length = ng1 * self.grid_width
        for i in range(ng1):
            Grid_coord.append(search_length * (-0.5 + (0.5 + i) / ng1))
        self.grid_coord = Grid_coord

        self.grid_r = [0 for _ in range(ng2 * ng1)]
        self.grid_i=[0 for _ in range(ng2 * ng1)]

        atom_coordinates=self.atom_coordinates
        grid_r=self.grid_r
        grid_i=self.grid_i
        # 创建线程池
        with ThreadPoolExecutor(max_workers=5) as executor:
            # 提交每个角度的计算任务给线程池
            futures = [executor.submit(self.process_rotation, ang,atom_coordinates,grid_r,grid_i) for ang in range(self.para.num_rot_angles)]

            # 等待所有任务完成
            for future in futures:
                future.result()

    def process_rotation(self, ang,atom_coordinates,grid_r,grid_i):
        thread_id = threading.current_thread().ident
        with self.lock:
            a=atom_coordinates[0][0][0]
            theta = ParamterStore.zangle[ang]  # 旋转角度
            self.ligand_rotation(theta)
            b=atom_coordinates[0][0][0]
            print(thread_id,'  ',a," ",self.atom_coordinates[0][0][0],end='\n')# 旋转配体
            # self.create_voxel()  # 计算得分
        # 执行其他操作，如配体的FFT操作


    def ligand_rotation(self,theta):
        r11 = np.cos(theta[0]) * np.cos(theta[2]) - np.sin(theta[0]) * np.cos(theta[1]) * np.sin(theta[2])
        r21 = np.sin(theta[0]) * np.cos(theta[2]) + np.cos(theta[0]) * np.cos(theta[1]) * np.sin(theta[2])
        r31 = np.sin(theta[1]) * np.sin(theta[2])
        r12 = -np.cos(theta[0]) * np.sin(theta[2]) - np.sin(theta[0]) * np.cos(theta[1]) * np.cos(theta[2])
        r22 = -np.sin(theta[0]) * np.sin(theta[2]) + np.cos(theta[0]) * np.cos(theta[1]) * np.cos(theta[2])
        r32 = np.sin(theta[1]) * np.cos(theta[2])
        r13 = np.sin(theta[0]) * np.sin(theta[1])
        r23 = -np.cos(theta[0]) * np.sin(theta[1])
        r33 = np.cos(theta[1])
        # 计算中心坐标不太一样
        for l in range(self.num_atoms):
            x=self.atom_coordinates[0][l][0]
            y=self.atom_coordinates[0][l][1]
            z=self.atom_coordinates[0][l][2]
            self.atom_coordinates[0][l][0]=r11 * x + r12 * y + r13 * z
            self.atom_coordinates[0][l][1]=r21 * x + r22 * y + r23 * z
            self.atom_coordinates[0][l][2]=r31 * x + r32 * y + r33 * z
    def rpscace_l(self,ACE_ratio, rec_core,ligand_core):
        delta=ligand_core
        surface = 1.0
        swollen_surface = -8888.0
        ng1=self.para.num_grid
        ng2=ng1*ng1
        ng3=ng1*ng2
        # 计算每个原子的表面、核心判断距离
        rcore2 = 1.5
        rsurf2 = 1.0
        for i in range(self.num_atoms):
            self.para.radius_core2.append(self.radius[i]*self.radius[i]*rcore2)
            self.para.radius_surf2.append(self.radius[i]*self.radius[i]*rsurf2)
        # Grid_coord = []
        # search_length=ng1*self.grid_width
        # for i in range(ng1):
        #     Grid_coord.append(search_length * (-0.5 + (0.5 + i) /ng1))
        # self.grid_coord = Grid_coord
        self.voxel_init()#计算每个原子与每个网格距离差的平方  好像差的稍微有点大
        # self.grid_r=[0 for _ in range(ng2*ng1)]
        for l in range(self.num_atoms):
            search_range = 2
            # 计算原子坐标在网格中的索引
            i2 = int(self.atom_coordinates[0][l][0] / self.grid_width + ng1 / 2)
            j2 = int(self.atom_coordinates[0][l][1] / self.grid_width + ng1 / 2)
            k2 = int(self.atom_coordinates[0][l][2] / self.grid_width + ng1 / 2)
            # 计算在网格搜索的上下界
            ia = max(i2 - search_range, 0)  # 下界
            ja = max(j2 - search_range, 0)  # 下界
            ka = max(k2 - search_range, 0)  # 下界
            ib = min(i2 + search_range + 1, ng1)  # 上界
            jb = min(j2 + search_range + 1, ng1)  # 上界
            kb = min(k2 + search_range + 1, ng1)  # 上界
            lc = ng1 * l

            # grid_r grid_i存储三维网格数据  应该是ng3的大小吧   初始化
            # self.grid_r = [[[0 for _ in range(ng1)] for _ in range(ng1)] for _ in range(ng1)]
            # self.grid_r=np.array(self.grid_r)

            # 从三个维度上搜索
            radius_core2_l = self.para.radius_core2[l]
            xd_lc = self.para.xd[lc + ia:lc + ib]
            yd_lc = self.para.yd[lc + ja:lc + jb]
            zd_lc = self.para.zd[lc + ka:lc + kb]
            # 优化后的循环
            for idx, i in enumerate(range(ia, ib)):
                if xd_lc[idx] > radius_core2_l:
                    continue
                for jdx, j in enumerate(range(ja, jb)):
                    d2 = xd_lc[idx] + yd_lc[jdx]
                    if d2 > radius_core2_l:
                        continue
                    for kdx, k in enumerate(range(ka, kb)):
                        d3 = d2 + zd_lc[kdx]
                        if d3 < radius_core2_l:
                            self.grid_r[ng2 * i + ng1 * j + k] = delta
        # num=0
        # for i in range(ng3):
        #     if self.grid_r[i]==delta:
        #         num+=1#差5个  差不多

        # scrape swollen surface
        for i in range(ng1):
            for j in range(ng1):
                for k in range(ng1):
                    ijk=ng2*i+ng1*j+k
                    if self.grid_r[ijk]==delta:
                        if self.grid_r[ijk-1]==0 or  self.grid_r[ijk+1]==0 or   self.grid_r[ijk-ng1]==0 or \
                        self.grid_r[ijk+ng1]==0 or  self.grid_r[ijk-ng2]==0  or  self.grid_r[ijk+ng2]==0 :
                            self.grid_r[ijk]=swollen_surface
        # num=0
        # for i in range(ng3):
        #     if self.grid_r[i]==swollen_surface:
        #         num+=1#7155

        for ijk in range(ng2,ng3-ng2):
            if self.grid_r[ijk]==swollen_surface:
                self.grid_r[ijk]=0.0

        # num=0
        # for i in range(ng3):
        #     if self.grid_r[i]==0.0:
        #         num+=1#四万8 差不多

        # make protein surface
        for l in range(self.num_atoms):
            search_range = 2
            # 计算原子坐标在网格中的索引
            i2 = int(self.atom_coordinates[0][l][0] / self.grid_width + ng1 / 2)
            j2 = int(self.atom_coordinates[0][l][1] / self.grid_width + ng1 / 2)
            k2 = int(self.atom_coordinates[0][l][2] / self.grid_width + ng1 / 2)
            # 计算在网格搜索的上下界
            ia = max(i2 - search_range, 0)  # 下界
            ja = max(j2 - search_range, 0)  # 下界
            ka = max(k2 - search_range, 0)  # 下界
            ib = min(i2 + search_range + 1, ng1)  # 上界
            jb = min(j2 + search_range + 1, ng1)  # 上界
            kb = min(k2 + search_range + 1, ng1)  # 上界
            lc = ng1 * l

            # grid_r grid_i存储三维网格数据  应该是ng3的大小吧   初始化
            # self.grid_r = [[[0 for _ in range(ng1)] for _ in range(ng1)] for _ in range(ng1)]
            # self.grid_r=np.array(self.grid_r)

            # 从三个维度上搜索
            radius_surfer2_l = self.para.radius_surf2[l]
            xd_lc = self.para.xd[lc + ia:lc + ib]
            yd_lc = self.para.yd[lc + ja:lc + jb]
            zd_lc = self.para.zd[lc + ka:lc + kb]
            # 优化后的循环
            for idx, i in enumerate(range(ia, ib)):
                if xd_lc[idx] > radius_surfer2_l :
                    continue
                for jdx, j in enumerate(range(ja, jb)):
                    d2 = xd_lc[idx] + yd_lc[jdx]
                    if d2 > radius_surfer2_l :
                        continue
                    for kdx, k in enumerate(range(ka, kb)):
                        d3 = d2 + zd_lc[kdx]
                        ijk=ng2 * i + ng1 * j + k
                        if self.grid_r[ijk]==delta:
                            continue
                        if d3 < radius_surfer2_l :
                            self.grid_r[ijk] =surface
        # num=0
        # for i in range(ng3):
        #     if self.grid_r[i]==surface:
        #         num+=1#一万2 差不多

        # 还有当delta不等于surf的值时的代码

    def electro_l(self,beta, elec_ratio):
        ng1=self.para.num_grid
        ng2=ng1**2
        ng3=ng1**3
        # self.grid_i=[0 for _ in range(ng3)]

        pad=(ng1*self.grid_width/2)
        for l in range(self.num_atoms):
            # 计算原子在网格中的位置
            i=max(0,min(ng1-1,(self.atom_coordinates[0][l][0]+pad)//self.grid_width))
            j=max(0,min(ng1-1,(self.atom_coordinates[0][l][1]+pad)//self.grid_width))
            k=max(0,min(ng1-1,(self.atom_coordinates[0][l][2]+pad)//self.grid_width))
            self.grid_i[int(i*ng2+j*ng1+k)]+=self.charge[l]

        # num=0
        # for i in range(ng3):
        #     if self.grid_i[i]!=0:
        #         num+=1  差不多














