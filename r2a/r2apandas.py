# -*- coding: utf-8 -*-
"""
@author: Hevelyn Sthefany L. Carvalho 
@author: Giordano Suffert Monteiro 
@author: Abel Augustu Alves Moreira
@date: 30/11/2020
@description: PyDash Project

An implementation example of a PANDAS Algorithm.

"""

from player.parser import *
from r2a.ir2a import IR2A
import numpy as np
import time
import statistics

class r2aPandas(IR2A):

    def __init__(self, id):
        IR2A.__init__(self, id)
        self.parsed_mpd = ''
        self.pandas = Pandas()

    def handle_xml_request(self, msg):
        self.trequest = time.perf_counter()
        self.pandas.update_request(time.perf_counter(), 0)
        self.send_down(msg)

    def handle_xml_response(self, msg):
        self.parsed_mpd = parse_mpd(msg.get_payload())
        self.pandas.qi = np.array(self.parsed_mpd.get_qi())

        if self.pandas.n == 0:  # faz a iniciação dos valores do algoritmo após receber o mpd
            self.pandas.initpandas(time.perf_counter(), msg.get_bit_length())
        self.send_up(msg)

    def handle_segment_size_request(self, msg):

        if self.pandas.n == 0: 
            buffer_size = 0
        else:
            buffer_size = self.whiteboard.get_playback_buffer_size()[-1][1]

        self.pandas.update_request(time.perf_counter(), buffer_size)
        msg.add_quality_id(self.pandas.get_quality())
        self.send_down(msg)

    def handle_segment_size_response(self, msg):
        self.pandas.update_response(time.perf_counter())
        self.send_up(msg)

    def initialize(self):
        pass

    def finalization(self):
        with open('../data.txt', 'wb')as f:
            f.write(("x:" + str(self.pandas.x)).encode())
            f.write(("\n\ny:" + str(self.pandas.y)).encode())
            f.write(("\n\nz:" + str(self.pandas.z)).encode())
            f.write(("\n\nz_estimado:" + str(self.pandas.estimated_z)).encode())
        
        print("qi:", self.pandas.qi)
        #pass
        


class Pandas:
    def __init__(self, w=0.3, k=0.14, beta=0.2, alfa=0.5, e=0.1, t=1, b=[], 
                 bmin=26, r = [], deltaup=0, deltadown=0, trequest=0, tresponse=0, 
                 n=-1, tnd=[], tr=[], td=[0], x=[], y=[], z=[], qi=np.array([])):
        self.trequest = trequest # momento em que foi realizado o request
        self.tresponse = tresponse # momento em que foi recebida a resposta
        self.n     = n     # numero do segmento
        self.w     = w     # taxa de bits de aumento aditivo de sondagem
        self.k     = k     # taxa de convergência de sondagem
        self.beta  = beta  # taxa de convergência do buffer do cliente
        self.alfa  = alfa  # Taxa de convergência de suavização
        self.e     = e     # margem de segurança multiplicativa
        self.t     = t     # duração do segmento de vídeo
        self.b     = b     # duração do buffer do cliente
        self.bmin  = bmin  # duração mínima do buffer do cliente
        self.r     = r     # taxa de bits de vídeo disponível em qi
        self.tnd   = tnd   # tempo alvo entre as solicitações
        self.tr    = tr    # tempo real entre as solicitações
        self.td    = td    # duração do download
        self.x     = x     # taxa média de dados alvo (ou compartilhamento de largura de banda)
        self.y     = y     # versão suavizada de x
        self.z     = z     # taxa de tranferência de TCP medida z=rt/T'
        self.qi    = qi    # Conjunto de taxas de bits de vídeo
        self.deltaup   = deltaup     # margem de segurança para cima 
        self.deltadown = deltadown   # margem de segurança para baixo

        self.avg_z = 0
        self.list_avg_z = []
        self.dev_z = 0
        self.estimated_z = []

    def initpandas(self, actual_tresponse, bit_length):
        #throughput do xml, primeiro throughput do algoritmo
        self.tresponse = actual_tresponse
        self.td[0] = actual_tresponse - self.trequest
        self.z.append(bit_length/self.td[0])
        self.list_avg_z.append(self.avg_z)
        idx_x0 = int(self.qi.size/2)   # escolher inicialização
        x0 = self.qi[idx_x0]  
        self.x.append(x0)
        self.y.append(x0)
        self.r.append(x0)
        self.tTarget_inter_request()

        
        #self.w = self.qi[0]
        self.w = self.qi[int(idx_x0/2)]
        #self.k = 0.7
        self.e = 0.1
        self.alfa = 0.5
        

    # Estimativa da porção da largura de banda
    
    def estimate_xn(self):
        self.tr.append(max(self.tnd[-1], self.td[-1])) #na primeira vez tr[0] = td[0]
        #self.b.append(max(0, self.b[-1] + self.t - self.tr[-1]))

        if (self.tnd[-1] > self.td[-1]):
            time.sleep(self.tnd[-1] - self.td[-1])

        m = max(0, self.x[-1]-self.z[-1]+self.w) 
        xn = self.x[-1] + self.k*self.tr[-1]*(self.w - m) 
        self.x.append(max(xn, self.qi[0])) #Valor mínimo para x
    
    def get_quality(self):
        self.estimate_xn()
        self.S()
        self.Q()

        #Tratamento de variação para a pior qualidade, pois não fica pior que a pior qualidade
        if (self.r[-2] != self.qi[0] and self.r[-1] == self.qi[0] and self.b[-1] > 0):
            qi2 = self.qi[self.qi <= self.b[-1] * self.estimated_z[-1]]
            qi2 = qi2[qi2 <= self.r[-2]]
            if(qi2.size): 
                self.r[-1] = qi2[-1]
                self.x[-1] = qi2[-1]
                self.y[-1] = qi2[-1]

        self.tTarget_inter_request()
        return self.r[-1]

    def S(self): #EWMA smoother
        yn = self.y[-1] - self.tr[-1] * self.alfa * (self.y[-1] - self.x[-1])
        self.y.append(max(yn, self.qi[0]))
    
    def Q(self): #dead-zone quantizer
        self.deltaup = self.e * self.y[-1]
        rup = self.get_rup()
        rdown = self.get_rdown()
        rn = 0
        if(self.r[-1] < rup): rn = rup 
        elif(rup <= self.r[-1] and self.r[-1] <= rdown): rn = self.r[-1] 
        else: rn = rdown 
        self.r.append(rn)
   
    def get_rup(self):
        y2 = self.y[-1] - self.deltaup
        qi2 = self.qi[self.qi <= y2] 
        if(qi2.size): return qi2[-1]
        else:         return self.qi[0]

    def get_rdown(self):
        y2 = self.y[-1] - self.deltadown 
        qi2 = self.qi[self.qi <= y2] 
        if(qi2.size): return qi2[-1]
        else:         return self.qi[0]


    def tTarget_inter_request(self):
        tnd = self.r[-1]*self.t/self.y[-1] + self.beta*(self.b[-1] - self.bmin)
        self.tnd.append(tnd)

    def update_response(self, actual_tresponse):
        self.tresponse = actual_tresponse
        self.td.append(self.tresponse - self.trequest) # tempo do download do segmento
        self.z.append((self.r[-1]*self.t)/self.td[-1]) # valor do throughput TCP real
    
        self.avg_z = self.avg_z*0.8 + self.z[-1]*0.2 #novo (0.125), alfa = 0.2
        #self.avg_z = 0.8*statistics.harmonic_mean([self.avg_z, self.z[1]])
        self.list_avg_z.append(self.avg_z)
        dev = self.z[-1] - self.avg_z
        if dev < 0: dev = dev * -1
        #self.dev_z = (0.75) * self.dev_z + 0.15 * dev #beta=0.15
        self.estimated_z.append(self.avg_z) #+ 4*self.dev_z

    def update_request(self, actual_trequest, buffer_size):
        self.n += 1
        self.trequest = actual_trequest
        self.b.append(buffer_size)
