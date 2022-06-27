import os
import random
from tqdm import tqdm
import neat
import numpy
from distutils.dir_util import copy_tree

MAX_SINGLE_TRAIN = 100
DISCRIMINATOR_EVAL_SET = 1000
GENERATOR_EVAL_CHANCES = 100
DATASET_FILENAME = "dataset.npy"

print("Building Populations")
gen_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,
						 "generator.cfg")
p_generator = neat.Population(gen_config)
p_generator.add_reporter(neat.StdOutReporter(False))
dis_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,
						 "discriminator.cfg")
p_discriminator = neat.Population(dis_config)
p_discriminator.add_reporter(neat.StdOutReporter(False))

print("loading dataset")
real_dataset = numpy.load(DATASET_FILENAME)
dataset_size = real_dataset.shape[0]
image_size = real_dataset.shape[1]
print("generating fake dataset")
fake_dataset = numpy.ndarray((dataset_size, image_size), dtype=numpy.float32)
net = neat.nn.FeedForwardNetwork.create(p_generator.population[1], gen_config)
for i in tqdm(range(dataset_size)):
	fake_dataset[i] = net.activate((random.random(),))
del net

training_cycle = 0


def eval_generator(genomes, config):
	for genome_id, genome in tqdm(genomes):
		g_net = neat.nn.FeedForwardNetwork.create(genome, config)
		# loss is discriminator output
		discr = neat.nn.FeedForwardNetwork.create(best_discr, dis_config)
		total_loss = 0
		for uwu in range(GENERATOR_EVAL_CHANCES):
			temp_gen_out = g_net.activate((random.random(),))
			total_loss += discr.activate(temp_gen_out)[0]
		genome.fitness = (GENERATOR_EVAL_CHANCES - total_loss) / GENERATOR_EVAL_CHANCES


def eval_discriminator(genomes, config):
	for genome_id, genome in tqdm(genomes):
		d_net = neat.nn.FeedForwardNetwork.create(genome, config)
		net_score = DISCRIMINATOR_EVAL_SET
		# evaluating for fakes, net result should be 1
		for owo in range(int(DISCRIMINATOR_EVAL_SET / 2)):
			out = d_net.activate(fake_dataset[random.randint(0, dataset_size - 1)])
			net_score -= (1 - out[0]) ** 2
		# evaluating for real, net result should be 0
		for uwo in range(int(DISCRIMINATOR_EVAL_SET / 2)):
			out = d_net.activate(real_dataset[random.randint(0, dataset_size - 1)])
			net_score -= (0 - out[0]) ** 2

		net_score /= DISCRIMINATOR_EVAL_SET
		genome.fitness = net_score


while True:
	print("Training Cycle n°:", training_cycle)
	p_generator.add_reporter(neat.Checkpointer(generation_interval=1, filename_prefix='networks/generator/cycle-'+str(training_cycle)+'-generator-'))
	p_discriminator.add_reporter(
		neat.Checkpointer(generation_interval=1, filename_prefix='networks/discriminator/cycle-'+str(training_cycle)+'-discriminator-'))
	copy_tree("networks/", "previous_networks/")
	os.rmdir("networks/discriminator")
	os.rmdir("networks/generator")
	os.mkdir("networks/discriminator")
	os.mkdir("networks/generator")
	# numpy.save("generated/"+str(training_cycle), fake_dataset)
	# Training discriminator
	best_discr = p_discriminator.run(eval_discriminator, 300)
	# Training generator
	best_gen = p_generator.run(eval_generator, 300)
	# Use best generator to create the new fake dataset
	best_generator = neat.nn.FeedForwardNetwork.create(best_gen, gen_config)
	for i in tqdm(range(dataset_size)):
		fake_dataset[i] = best_generator.activate((random.random(),))
	training_cycle += 1
