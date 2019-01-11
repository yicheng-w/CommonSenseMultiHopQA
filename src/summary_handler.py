import tensorflow as tf
import numpy as np


class SummaryHandler(object):
	
	def __init__(self,summary_file,summary_list):
		self.writer = tf.summary.FileWriter(summary_file)
		self.summary_graphs,self.summary_placeholders = self.prepare_summaries(summary_list)


	def prepare_summaries(self,summary_list):
		summary_graphs = {}
		summary_placeholders = {}
		for summary_type in summary_list:
			summary_placeholders[summary_type] = tf.placeholder(tf.float32)
			summary_graphs[summary_type] = tf.summary.scalar(summary_type,summary_placeholders[summary_type])	

		return summary_graphs,summary_placeholders

	def write_summaries(self,sess,scores):

		for key,value in self.summary_graphs.iteritems():
			if scores.get(key) is not None:
				summary = sess.run(value,feed_dict={self.summary_placeholders[key]:scores[key]})
				self.writer.add_summary(summary,scores['ITERATION'])

		self.writer.flush()


	def close_writer(self):
		self.writer.close()



