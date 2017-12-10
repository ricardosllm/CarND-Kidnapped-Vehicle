#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  num_particles = 100;
  double std_x   = std[0];
  double std_y   = std[1];
  double std_psi = std[2];

  // Create a normal Gaussian distribution
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_psi(theta, std_psi);

  default_random_engine gen;

  // Initialize particles
  for (int i = 0; i < num_particles; i++) {
    Particle particle = { i, dist_x(gen), dist_y(gen), dist_psi(gen) };

    particles.push_back(particle);
    weights.push_back(1);
  }

  // Complete initialization
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t,  double std_pos[],
                                double velocity, double yaw_rate) {
  default_random_engine gen;
  double x, y, theta;

  for (auto i = particles.begin(); i < particles.end(); i++) {
    Particle& particle = *i;

    x     = particle.x + velocity * cos(particle.theta) * delta_t;
    y     = particle.y + velocity * sin(particle.theta) * delta_t;
    theta = particle.theta + yaw_rate * delta_t;

    normal_distribution<double> dist_x(x, std_pos[0]);
    normal_distribution<double> dist_y(y, std_pos[1]);
    normal_distribution<double> dist_psi(theta, std_pos[2]);

    particle.x     = dist_x(gen);
    particle.y     = dist_y(gen);
    particle.theta = dist_psi(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
                                     std::vector<LandmarkObs>& observations) {
  for (auto i_obsrv = observations.begin(); i_obsrv < observations.end(); i_obsrv++) {
    LandmarkObs& observation = *i_obsrv;
    double final_dist        = INFINITY;

    for (auto i_pred = predicted.begin(); i_pred < predicted.end(); i_pred++) {
      LandmarkObs& pred = *i_pred;
      double dist = sqrt(pow(observation.x - pred.x, 2) + pow(observation.y - pred.y, 2));

      if (dist < final_dist) {
        final_dist = dist;
        observation.id = i_pred - predicted.begin();
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations,
                                   Map map_landmarks) {
  double sum_weights = 0;

  for (auto i_particle = particles.begin(); i_particle < particles.end(); i_particle++) {
    std::vector<int>    associations_id;
    std::vector<double> sense_x;
    std::vector<double> sense_y;

    Particle& particle = *i_particle;
    std::vector<LandmarkObs> predicted;

    for (auto i_landmark = map_landmarks.landmark_list.begin();
         i_landmark < map_landmarks.landmark_list.end(); i_landmark++) {
      Map::single_landmark_s& landmark = *i_landmark;

      double dist = sqrt(pow(landmark.x_f - particle.x, 2) + pow(landmark.y_f - particle.y, 2));

      if (dist < sensor_range) {
        associations_id.push_back(landmark.id_i);
        sense_x.push_back(landmark.x_f);
        sense_y.push_back(landmark.y_f);

        LandmarkObs predicted_landmark = glob2particle(landmark, particle);
        predicted.push_back(predicted_landmark);
      }
    }

    particle = SetAssociations(particle, associations_id, sense_x, sense_y);
    dataAssociation(predicted, observations);

    particle.weight = 1;
    for (auto i_obsrv = observations.begin(); i_obsrv < observations.end(); i_obsrv++) {
      LandmarkObs& observation = *i_obsrv;

      double delta_x = predicted[observation.id].x - observation.x;
      double delta_y = predicted[observation.id].y - observation.y;
      double argu    = pow(delta_x * std_landmark[0], 2) + pow(delta_y * std_landmark[1], 2);

      particle.weight *= exp(-0.5 * argu) / 2 / M_PI / std_landmark[0] / std_landmark[1];
    }

    weights[i_particle - particles.begin()] = particle.weight;
    sum_weights += particle.weight;
  }

  // Normalize the weights
  for (auto i_particle = particles.begin(); i_particle < particles.end(); i_particle++) {
    Particle& particle = *i_particle;

    particle.weight /= sum_weights;
    weights[i_particle - particles.begin()] /= sum_weights;
  }
}

void ParticleFilter::resample() {
  discrete_distribution<int> resampler(weights.begin(), weights.end());
  vector<Particle> new_particles;
  default_random_engine gen;

  for (int i = 0; i < num_particles; i++) {
    int n = resampler(gen);
    new_particles.push_back(particles[n]);
  }

  particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle,
                                         std::vector<int> associations,
                                         std::vector<double> sense_x,
                                         std::vector<double> sense_y) {
  // Clear associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations = associations;
  particle.sense_x      = sense_x;
  particle.sense_y      = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best) {
	vector<int> v = best.associations;
	stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space

  return s;
}

string ParticleFilter::getSenseX(Particle best) {
	vector<double> v = best.sense_x;
	stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space

  return s;
}

string ParticleFilter::getSenseY(Particle best) {
	vector<double> v = best.sense_y;
	stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space

  return s;
}

LandmarkObs ParticleFilter::glob2particle(Map::single_landmark_s g_landmark, Particle particle) {
  LandmarkObs landmark_observations;

  Eigen::MatrixXd R_c2g(2,2);
  Eigen::MatrixXd t_c2g(2,1);
  Eigen::MatrixXd landmark_g(2,1);

  R_c2g      << cos(particle.theta), -sin(particle.theta),
                sin(particle.theta), cos(particle.theta);

  t_c2g      << particle.x, particle.y;

  landmark_g << g_landmark.x_f, g_landmark.y_f;

  Eigen::MatrixXd R_c2g_transp = R_c2g.transpose();
  Eigen::MatrixXd obs = R_c2g_transp * landmark_g - R_c2g_transp * t_c2g;

  landmark_observations.id = g_landmark.id_i;
  landmark_observations.x  = obs(0,0);
  landmark_observations.y  = obs(1,0);

  return landmark_observations;
}
