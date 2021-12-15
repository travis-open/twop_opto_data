import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import math


def assign_est_cnx_prob_2d(df, x_bins, y_bins, prob_matrix, x_column='abs_x', y_column='y_pia'): ##assign each tested connection a prob. using prob_matrix. Used in conjunction with cnx_prob_2d to assign each tested connection the average Pcnx measured in the population
    y_len, x_len=np.shape(prob_matrix)
    for x in range(0,x_len-1):
        start_x=x_bins[x]
        stop_x=x_bins[x+1]
        for y in range(0,y_len-1):
            start_y=y_bins[y]
            stop_y=y_bins[y+1]
            prob=prob_matrix[y,x]
            df.loc[(df[x_column]>start_x) & (df[x_column]<stop_x) & (df[y_column]>start_y) & (df[y_column]<stop_y), 'est_cnx_prob'] = prob
    return df


def clean_axes(ax, fs):
	tick_size=4
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	
def cnx_prob_2d(df, cnx_calls, x_bins, y_bins, min_probed,x_column='abs_x',y_column='y_pia'): ##generate a 2d array with connection probabilities measured in df, elements with tested connections < min_probed set to nan
    df=df.dropna(subset=[x_column,y_column])
    probed_calls=['no cnx', 'excitatory', 'inhibitory', 'tbd excitatory', 'tbd inhibitory', 'tbd latency']
    probed_df=df[df['cnx'].isin(probed_calls)]
    cnx_df=df[df['cnx'].isin(cnx_calls)]
    H, xedges, yedges = np.histogram2d(probed_df[x_column], probed_df[y_column], bins=(x_bins, y_bins))
    threshold=H >min_probed
    H=H*threshold
    H = H.T
    H_cnx, xedges, yedges = np.histogram2d(cnx_df[x_column],cnx_df[y_column], bins=(x_bins, y_bins))
    H_cnx=H_cnx*threshold
    H_cnx = H_cnx.T
    H_prob=H_cnx/H
    return H_prob


def connectivity_profile(df, column_name, bins):
	probed_calls=['no cnx', 'excitatory', 'inhibitory', 'tbd excitatory', 'tbd inhibitory', 'tbd latency']
	probed_hist, x=hist_of_call(df, probed_calls, column_name, bins)
	exc_hist, x=hist_of_call(df, 'excitatory', column_name, bins)
	direct_hist, x=hist_of_call(df, 'direct', column_name, bins)
	inh_hist, x=hist_of_call(df, 'inhibitory', column_name, bins)
	exc_frac=exc_hist.astype(float)/probed_hist.astype(float)
	exc_ci=sm.stats.proportion_confint(exc_hist.astype(float),probed_hist.astype(float), method='jeffreys')
	direct_frac=direct_hist.astype(float)/probed_hist.astype(float)
	direct_ci=sm.stats.proportion_confint(direct_hist.astype(float),probed_hist.astype(float), method='jeffreys')
	inh_frac=inh_hist.astype(float)/probed_hist.astype(float)
	inh_ci=sm.stats.proportion_confint(inh_hist.astype(float),probed_hist.astype(float), method='jeffreys')
	return {'midX':midPoints(x),'probed_hist':probed_hist,'exc_hist':exc_hist,'inh_hist':inh_hist,'direct':direct_hist, 
	'exc_frac':exc_frac,'exc_ci':exc_ci, 'direct_frac':direct_frac, 'direct_ci':direct_ci, 'inh_frac':inh_frac,'inh_ci':inh_ci,'bins':bins}

def converge_params(name, group, i, j):
    prepre_dx=group.x_rotate[j]-group.x_rotate[i]
    prepre_dy=group.y_rotate[j]-group.y_rotate[i]
    prepre_dz=group.z_pos[j]-group.z_pos[i]
    prepre_dxyz=math.sqrt(prepre_dx**2+prepre_dy**2+prepre_dz**2)
    xyz_dist_i=group.xyz_dist[i]
    xyz_dist_j=group.xyz_dist[j]
    x_ap_i=group.x_ap[i]
    x_ap_j=group.x_ap[j]
    pia_i=group.y_pia[i]
    pia_j=group.y_pia[j]
    amp_i=group.peakOfAvg[i]
    amp_j=group.peakOfAvg[j]
    cnx_i=group.cnx[i]
    cnx_j=group.cnx[j]
    MP_i=group.MP_ID[i]
    MP_j=group.MP_ID[j]
    post_layer=group.post_layer[i]
    post_class=group.post_class[i]
    post_pia=group.toPia[i]
    post_depth=group.post_depth[i]
    pre_depth_i=group.pre_depth[i]
    pre_depth_j=group.pre_depth[j]
    conv_prob=group.est_cnx_prob[i]*group.est_cnx_prob[j]
    conv_dict={'name':name,'MP_i':MP_i,'MP_j':MP_j,'prepre_dx':prepre_dx,'prepre_dy':prepre_dy,
               'prepre_dz':prepre_dz,'prepre_dxyz':prepre_dxyz,'pia_i':pia_i,'pia_j':pia_j,
               'x_ap_i':x_ap_i,'x_ap_j':x_ap_j,'amp_i':amp_i,'amp_j':amp_j,'cnx_i':cnx_i, 'cnx_j':cnx_j,
               'post_pia':post_pia, 'post_layer':post_layer, 'post_class':post_class,'conv_prob':conv_prob,
               'post_depth_i':post_depth, 'pre_depth_i':pre_depth_i, 'pre_depth_j':pre_depth_j, 
               'xyz_dist_i':xyz_dist_i, 'xyz_dist_j':xyz_dist_j}
    return conv_dict


def diverge_params(name, group, i, j):
    postpost_dx=abs(group.x_rotate[j]-group.x_rotate[i])
    postpost_dy=abs(group.y_rotate[j]-group.y_rotate[i])
    postpost_dz=abs(group.HS_z[j]-group.HS_z[i])
    postpost_dxyz=math.sqrt(postpost_dx**2+postpost_dy**2+postpost_dz**2)
    prepost_dx=(group.abs_x[i]+group.abs_x[j])/2
    prepost_dy=(group.abs_y[i]+group.abs_y[j])/2
    xyz_dist_i=group.xyz_dist[i]
    xyz_dist_j=group.xyz_dist[j]
    x_ap_i=group.x_ap[i]
    x_ap_j=group.x_ap[j]
    pia_i=group.toPia[i]
    pia_j=group.toPia[j]
    amp_i=group.peakOfAvg[i]
    amp_j=group.peakOfAvg[j]
    cnx_i=group.cnx[i]
    cnx_j=group.cnx[j]
    cnx_to_i=group.cnx_to[i]
    cnx_to_j=group.cnx_to[j]
    HS_i=group.headstage[i]
    HS_j=group.headstage[j]
    post_class_i=group.post_class[i]
    post_class_j=group.post_class[j]
    post_layer_i=group.post_layer[i]
    post_layer_j=group.post_layer[j]
    pre_pia=(group.y_pia[i]+group.y_pia[j])/2
    pre_depth=(group.pre_depth[i]+group.pre_depth[j])/2
    post_depth_i=group.post_depth[i]
    post_depth_j=group.post_depth[j]
    div_prob=group.est_cnx_prob[i]*group.est_cnx_prob[j]
    div_dict={'name':name,'HS_i':HS_i,'HS_j':HS_j,'postpost_dx':postpost_dx,'postpost_dy':postpost_dy,'postpost_dz':postpost_dz,'postpost_dxyz':postpost_dxyz,
             'prepost_dx':prepost_dx,'prepost_dy':prepost_dy,'pia_i':pia_i,'pia_j':pia_j,'x_ap_i':x_ap_i,'x_ap_j':x_ap_j,
              'amp_i':amp_i,'amp_j':amp_j, 'cnx_i':cnx_i, 'cnx_j':cnx_j,'pre_pia':pre_pia, 'post_class_i':post_class_i, 'post_class_j':post_class_j,
             'post_layer_i':post_layer_i,'post_layer_j':post_layer_j,'cnx_to_i':cnx_to_i,'cnx_to_j':cnx_to_j, 'div_prob':div_prob,
             'pre_depth':pre_depth, 'post_depth_i':post_depth_i, 'post_depth_j':post_depth_j, 'xyz_dist_i':xyz_dist_i, 'xyz_dist_j':xyz_dist_j}
    return div_dict

##used in convergence/divergence analysis frac_ci_j_give_i returns observed likelihood of completed motif and associated 95% CI's
def frac_ci_j_give_i(df,param,cnx_call, bins):
    cnx_i=df[df.cnx_i.isin(cnx_call)]
    conv=cnx_i[cnx_i.cnx_j.isin(cnx_call)]
    probed_hist=np.histogram(cnx_i[param], bins=bins)[0]
    conv_hist=np.histogram(conv[param], bins=bins)[0]
    frac=conv_hist.astype(float)/probed_hist
    ci=sm.stats.proportion_confint(conv_hist.astype(float),probed_hist.astype(float), method='jeffreys')
    return {'frac':frac, 'ci':ci, 'probed':probed_hist,'conv':conv_hist}

def graph_map_subplot(ax, df, cnx_hit, pre_color, pre_marker, post_color, post_marker, post_size=75, pre_size=50):
	nc_calls=['no cnx', 'tbd excitatory', 'tbd inhibitory', 'tbd latency']
	no_cnx=df[df.cnx.isin(nc_calls)]
	if type(cnx_hit) is str:
		cnx_hit=[cnx_hit]
	cnx=df[df.cnx.isin(cnx_hit)]
	ax.scatter(no_cnx['x_ap'], no_cnx['y_pia'], color='grey', alpha=0.5, facecolor='white', s=pre_size, marker=pre_marker,zorder=0)
	ax.scatter(cnx['x_ap'], cnx['y_pia'], facecolor=pre_color, color='white',alpha=0.5, s=pre_size, marker=pre_marker,zorder=2)
	cells=df.groupby(['exp_id', 'headstage']).toPia.mean()
	for cell in cells:
		alpha=0.3
		if len(cells)>10:
			alpha=0.1
		ax.scatter(0, cell, marker=post_marker, color=post_color, alpha=alpha, s=post_size,zorder=2)

def graph_map_subplot_color_dict(ax, df, cnx_hit, color_dict, pre_marker, post_color, post_marker, post_size=75, pre_size=50):
	nc_calls=['no cnx', 'tbd excitatory', 'tbd inhibitory', 'tbd latency']
	no_cnx=df[df.cnx.isin(nc_calls)]
	if type(cnx_hit) is str:
		cnx_hit=[cnx_hit]
	cnx=df[df.cnx.isin(cnx_hit)]
	ax.scatter(no_cnx['x_ap'], no_cnx['y_pia'], color='grey', alpha=0.2, facecolor='grey',edgecolor='none', s=pre_size, marker=pre_marker,zorder=0)
	ax.scatter(cnx['x_ap'], cnx['y_pia'], facecolor=[color_dict[i] for i in cnx.presynapticCre], edgecolor='white', alpha=0.5, s=pre_size, marker=pre_marker,zorder=1)
	cells=df.groupby(['exp_id', 'headstage']).toPia.mean()
	ax.scatter(0,np.nanmean(cells),marker=post_marker,color=post_color,alpha=0.5,s=post_size,zorder=2)

def graph_selected(ax, df, cnx_hit, pre_color, pre_marker, pre_size=75): ##eg direct artifact
	select_df=df[df.cnx==cnx_hit]
	ax.scatter(select_df['x_ap'], select_df['y_pia'], color=pre_color, alpha=0.5, s=pre_size, marker=pre_marker,zorder=0)

def hist_of_call(df, call, column_name, bins):
	if type(call)==str:
		call=[call]
	values=df[df['cnx'].isin(call)][column_name]
	hist, x=np.histogram(values, bins=bins)
	return hist, x


def layer_borders(ax,max_x=700,borders=[80,330,480,750,1000], lw=1.2, c='black', alpha=1, ls='--'):  #dim indicates x or y
	x_vals=[-1*max_x,max_x]
	for bord in borders:
		y_vals=[bord,bord]
		ax.plot(x_vals,y_vals, ls=ls, lw=lw, c=c, alpha=alpha)

def layer_borders_x(ax,max_y=700,borders=[80,330,480,750,1000], lw=1.2, c='black', alpha=1, ls='--'):  #dim indicates x or y
	y_vals=[-1*max_y,max_y]
	for bord in borders:
		x_vals=[bord,bord]
		ax.plot(x_vals,y_vals, ls=ls, lw=lw, c=c,alpha=alpha)


def make_ordered_conv_df(df):
    grouped=df.groupby(['exp_id', 'headstage'], sort=False)
    rows=[]
    for name, group in grouped:
        group=group[group.cnx.isin(['no cnx','inhibitory','excitatory','tbd inhibitory', 'tbd excitatory', 'tbd latency'])]
        group=group.reset_index()
        for i in group.index:
            for j in group.index:
                if i!=j:
                    conv_dict=converge_params(name,group,i,j)
                    rows.append(conv_dict)        
    conv_df=pd.DataFrame.from_dict(rows)
    return conv_df


def make_ordered_div_df(df):
    grouped=df.groupby(['exp_id', 'MP_ID'], sort=False)
    rows=[]
    for name, group in grouped:
        group=group[group.cnx.isin(['no cnx','inhibitory','excitatory','tbd inhibitory', 'tbd excitatory', 'tbd latency'])]
        group=group.reset_index()
        for i in group.index:
            for j in group.index:
                if i!=j:
                    div_dict=diverge_params(name,group,i,j)
                    rows.append(div_dict)
    div_df=pd.DataFrame.from_dict(rows)
    return div_df

def midPoints(x):
	x=np.array(x)
	x_values=(x[:-1]+x[1:])/2
	return x_values

def plot_cdf(ax, values,color='black',style='solid'):

    v_sorted= np.sort(values)
    cdf=np.linspace(0.,1.,np.size(v_sorted))
    ax.plot(v_sorted,cdf,c=color, ls=style, lw=2)


def plotFrac_wHist2(probed1,cnx1,label1,color1,probed2,cnx2,label2,color2,xlabel, bins, min_probed=5,ymax=1): ##plot two datasets on stacked axes		
	fs=10
	tick_size=5
	font='arial'
	summary_dict={}
	probed_hist1, x=np.histogram(probed1, bins=bins)
	called_hist1, x=np.histogram(cnx1,bins=bins)
	print ("probed 1: ", probed_hist1)
	print ("called 1: ", called_hist1)
	anno1=called_hist1
	probed_hist2, x=np.histogram(probed2,bins=bins)
	called_hist2, x=np.histogram(cnx2,bins=bins)
	print ("probed 2: ", probed_hist2)
	print ("called 2: ", called_hist2)
	summary_dict['bins']=bins
	summary_dict['probed_hist1']=probed_hist1
	summary_dict['cnx_hist1']=called_hist1
	summary_dict['probed_hist2']=probed_hist2
	summary_dict['cnx_hist2']=called_hist2
	anno2=called_hist2
	threshold1=probed_hist1>min_probed
	threshold2=probed_hist2>min_probed
	probed_hist1=probed_hist1[threshold1]
	probed_hist2=probed_hist2[threshold2]
	called_hist1=called_hist1[threshold1]
	called_hist2=called_hist2[threshold2]
	frac1=called_hist1.astype(float)/probed_hist1.astype(float)
	frac2=called_hist2.astype(float)/probed_hist2.astype(float)
	summary_dict['cnx_frac1']=frac1
	summary_dict['cnx_frac2']=frac2
	ci1=sm.stats.proportion_confint(called_hist1.astype(float),probed_hist1.astype(float), method='jeffreys')
	ci2=sm.stats.proportion_confint(called_hist2.astype(float),probed_hist2.astype(float), method='jeffreys')
	summary_dict['ci1']=ci1
	summary_dict['ci2']=ci2
	midX=midPoints(bins)
	fig, axes = plt.subplots(3,1,figsize=(2.5,3), sharex=True, gridspec_kw={'height_ratios':[6,2,2]})
	ax = axes[0]
	plotRateCI(ax,midX[threshold1],frac1,ci1,color1)
	plotRateCI(ax,midX[threshold2],frac2,ci2,color2)
	ax.set_ylim([0,ymax])
	
	clean_axes(ax,fs)
	ax.set_ylabel('fraction connected', fontsize=fs, fontname=font)
	ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
	ax.yaxis.set_tick_params(size=tick_size,labelsize=fs, direction='out')
	ax.yaxis.set_major_locator(plt.MaxNLocator(4))
	
	ax.xaxis.set_ticks_position('none')
	
	
	ax=axes[1]

	ax.hist(probed1,bins=bins, color=color1, alpha=0.5)
	ax.hist(cnx1, bins=bins, color=color1)
	clean_axes(ax,fs)
	
	ax.set_ylabel(label1, fontsize=fs, fontname=font)

	ax.yaxis.set_major_locator(plt.MaxNLocator(2))
	ax.yaxis.set_tick_params(size=tick_size,labelsize=fs, direction='out')
	ax.xaxis.set_ticks_position('none')
	
	
	ax=axes[2]
	ax.hist(probed2,bins=bins, color=color2, alpha=0.5)
	ax.hist(cnx2,bins=bins,color=color2)
	ax.set_ylabel(label2, fontsize=fs, fontname=font)
	ax.xaxis.set_ticks_position('bottom')
	clean_axes(ax,fs)
	ax.yaxis.set_major_locator(plt.MaxNLocator(2))
	ax.xaxis.set_major_locator(plt.MaxNLocator(4))
	ax.set_xlim([0,bins[-1]])
	ax.set_xlabel(xlabel, size=fs)
	ax.yaxis.set_tick_params(size=tick_size,labelsize=fs, direction='out')
	ax.xaxis.set_tick_params(size=tick_size,labelsize=fs, direction='out')
	ax.set_xlim([0,bins[-1]+1])
	#fig.text(-0.15,0.38, 'connections probed', rotation='vertical', fontsize=fs, fontname=font)
	fig.set_figwidth(1.75)
	return summary_dict


def plotRateCI(ax,midX,rate,ci, color,alpha=0.3, ls='solid'):
	if len(midX)>1:
		ax.plot(midX, rate, c=color, lw=2,marker='s', markersize=3,alpha=0.8, ls=ls)
		ax.fill_between(midX,ci[0],ci[1],color=color, alpha=alpha)
	else:
		ax.plot(midX, rate, c=color, lw=2, marker='s', markersize=3,alpha=0.8, ls=ls)
		broadX=[midX[0]-15,midX[0]+15]
		ci_l=np.repeat(ci[0],2)
		ci_h=np.repeat(ci[1],2)
		ax.fill_between(broadX,ci_l,ci_h,color=color,alpha=alpha)

def plotRateCIy(ax,midX,rate,ci, color,ls='solid'):
	if len(midX)>1:
		ax.plot(rate, midX, c=color, lw=2,marker='s', alpha=0.8, markersize=4,ls=ls)
		ax.fill_betweenx(midX,ci[0],ci[1],color=color, alpha=0.3)
	else:
		ax.plot(rate, midX, c=color, lw=2, marker='s', alpha=0.8, markersize=4,ls=ls
			)
		broadX=[midX[0]-50,midX[0]+50]
		ci_l=np.repeat(ci[0],2)
		ci_h=np.repeat(ci[1],2)
		ax.fill_betweenx(broadX,ci_l,ci_h,color=color,alpha=0.3)


def print_counts(df):
	df=df.dropna(subset=['y_pia'])
	probed_calls=['no cnx', 'excitatory', 'inhibitory', 'tbd excitatory', 'tbd inhibitory', 'tbd latency']
	probed_df=df[df.cnx.isin(probed_calls)]
	exc_df=df[df.cnx=='excitatory']
	inh_df=df[df.cnx=='inhibitory']
	print (np.shape(probed_df)[0], " probed cnxs")
	print (np.shape(exc_df)[0], " exc cnxs")
	print (np.shape(inh_df)[0], " inh cnxs")
	grouped=df.groupby(['exp_id','headstage'])
	count=0
	for name, group in grouped:
		count+=1
	print (count, "postsyanptic cells")


def rate_ci_probed_cnx_x(ax, color, midX, probed_hist, cnx_hist, min_probed=5,alpha=0.3, ls='solid'):
	threshold=probed_hist>min_probed
	probed_hist=probed_hist[threshold]
	cnx_hist=cnx_hist[threshold]
	midX=midX[threshold]
	frac=cnx_hist.astype(float)/probed_hist.astype(float)
	ci=sm.stats.proportion_confint(cnx_hist.astype(float),probed_hist.astype(float), method='jeffreys')
	plotRateCI(ax, midX, frac, ci, color,alpha,ls)

def rate_ci_probed_cnx_y(ax, color, midX, probed_hist, cnx_hist, min_probed=5,alpha=0.3, ls='solid'):
	threshold=probed_hist>min_probed
	probed_hist=probed_hist[threshold]
	cnx_hist=cnx_hist[threshold]
	midX=midX[threshold]
	frac=cnx_hist.astype(float)/probed_hist.astype(float)
	ci=sm.stats.proportion_confint(cnx_hist.astype(float),probed_hist.astype(float), method='jeffreys')
	plotRateCIy(ax, midX, frac, ci, color,ls)


