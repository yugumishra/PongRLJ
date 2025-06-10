package environment;
import org.lwjgl.glfw.GLFW;

import ann.*;

public class Pong {
	Drawable board1;
	Drawable board2;
	Drawable ball;
	
	float ballXVelocity;
	float ballYVelocity;
	
	float board1Velocity;
	float board2Velocity;
	
	Drawable[] divider;
	
	int playerOnePoints;
	int playerTwoPoints;
	
	Window w;
	
	int frameCounter = 50;
	int numGames;
	
	// ---------- ANN variables ---------------
	public static int MAX_EPISODES = 100_000;
	public static int datasetSize = 10_000;
	public static int STATE_VECTOR_LENGTH = 8;
	public static int NUM_ACTIONS = 3;
	public static int HIDDEN_SPACE_LENGTH = 32;

	Tensor states;
	Tensor predictedQs;
	int[] actions;
	Tensor rewards;
	Tensor nextStates;
	boolean[] terminals;

	Ann ann;
	Ann targetNet;

	int networkUpdates = 0;
	int experienceCounter = 0;

	int nnPoints;
	int totPoints;

	float eps = 1.0f;
	float epsDecay = 0.99f;
	float gamma = 0.99f;

	Optimizer optimizer;
	Metrics tracker;

	boolean fast = true;
	
	public Pong() {
		//init the drawables
		//assume 800x800 screen
		board1 = new Drawable();
		board1.xDisplacement = -0.85f;
		float[] vertices = {
			-0.025f,  0.125f,
			-0.025f, -0.125f,
			 0.025f, -0.125f,
			 
			-0.025f,  0.125f,
			 0.025f,  0.125f,
			 0.025f, -0.125f,
		};
		board1.shape = vertices;
		
		board1.load();
		
		
		board2 = new Drawable();
		board2.xDisplacement = 0.85f;
		float[] vertices2 = {
			-0.025f,  0.125f,
			-0.025f, -0.125f,
			 0.025f, -0.125f,
			 
			-0.025f,  0.125f,
			 0.025f,  0.125f,
			 0.025f, -0.125f,
		};
		board2.shape = vertices2;
		board2.load();
		
		ball = new Drawable();
		float[] vertices3 = {
			-0.025f,  0.025f,
			-0.025f, -0.025f,
			 0.025f, -0.025f,
			 
			-0.025f,  0.025f,
			 0.025f,  0.025f,
			 0.025f, -0.025f,
		};
		ball.shape = vertices3;
		
		ball.load();
		
		divider = new Drawable[35];
		for(int i= 0; i< 35; i++) {
			Drawable d = new Drawable();
			float[] borderVertices = {
				-0.01f,  0.01f,
				-0.01f, -0.01f,
				 0.01f, -0.01f,
				 
				-0.01f,  0.01f,
				 0.01f,  0.01f,
				 0.01f, -0.01f
			};
			d.shape = borderVertices;
			d.load();
			d.yDisplacement = 0.98f - 0.06f * i;
			divider[i] = d;
		}
		
		ballXVelocity = -0.02f;
		ballYVelocity =  0.0f;
		
		//init anns
		Input input = new Input(new Shape(STATE_VECTOR_LENGTH));
		
		Dense hidden1 = new Dense(input, new Shape(HIDDEN_SPACE_LENGTH), false);
		Activation activation1 = new Activation(hidden1, ActivationFunction.RELU);
		Dropout drop1 = new Dropout(activation1, 0.2f);

		Dense hidden2 = new Dense(drop1, new Shape(HIDDEN_SPACE_LENGTH), false);
		Activation activation2 = new Activation(hidden2, ActivationFunction.RELU);
		Dropout drop2 = new Dropout(activation2, 0.5f);
		
		Dense output = new Dense(drop2, new Shape(NUM_ACTIONS), true);
		ann = new Ann(input, output);
		
		ann.printSummary();
		
		//send 0s through the network for initialization
		Tensor t = new Tensor(new Shape(8, STATE_VECTOR_LENGTH));
		t.init();
		ann.predict(t);

		//initialize optimizer and tracker
		optimizer = new AdamOptimizer(3e-4f, 0.995f, 0.9f, 0.999f, 0.0f);
		tracker = new ann.Metrics(true, false, true);

		//save network into target file, then read it for the target network
		
		ann.save("targetNet_" + networkUpdates + ".bin");
		targetNet = Ann.load("targetNet_"+ networkUpdates + ".bin");

		//initialize storage
		states = new Tensor(new Shape(datasetSize, STATE_VECTOR_LENGTH));
		predictedQs = new Tensor(new Shape(datasetSize, NUM_ACTIONS));
		nextStates = new Tensor(new Shape(datasetSize, STATE_VECTOR_LENGTH));
		states.init();
		predictedQs.init();
		nextStates.init();

		rewards = new Tensor(new Shape(datasetSize));
		rewards.init();

		actions = new int[datasetSize];
		terminals = new boolean[datasetSize];

		nnPoints = 0;
		totPoints = 0;
	}
	
	public void play() {
		//loop
		while(true) {
			if(!fast) {
				w.clear();
				w.draw(board1);
				w.draw(board2);
				if(frameCounter == 50) w.draw(ball);
				for(Drawable d: divider) w.draw(d);
				w.update();
			}
			
			update();
			
			if(w.shouldClose() || numGames == MAX_EPISODES) break;
			
		}
		w.cleanup();
		
		board1.cleanup();
		board2.cleanup();
		ball.cleanup();
		for(Drawable d: divider) d.cleanup();
	}

	public void update() {
		Tensor state = getState();
		int action = getAnnAction(state);
		//update buffers
		for(int i= 0; i< STATE_VECTOR_LENGTH; i++) {
			states.set(STATE_VECTOR_LENGTH * experienceCounter + i, state.at(i));
		}
		actions[experienceCounter] = action;

	  if(action == 0 || w.getKey(GLFW.GLFW_KEY_W)) { 
	      board1.yDisplacement += 0.03f;
	      board1Velocity = 0.03f;
	      
	  } else if(action == 1 || w.getKey(GLFW.GLFW_KEY_S)) { 
	      board1.yDisplacement -= 0.03f;
	      board1Velocity = 0.03f;
	      
	  }
	    
	  //simple board2 policy
	  float displacement = 0.075f * (ball.yDisplacement - board2.yDisplacement);
	  board2Velocity = Math.signum(displacement) * Math.min(Math.abs(displacement), 0.03f);
	  board2.yDisplacement = board2.yDisplacement + board2Velocity;
	    
	    
		//propagating ann action to next state
		board1.yDisplacement = Math.min(board1.yDisplacement, 1.0f - 0.125f);
		board1.yDisplacement = Math.max(board1.yDisplacement, -1.0f + 0.125f);
		
		board2.yDisplacement = Math.min(board2.yDisplacement, 1.0f - 0.125f);
		board2.yDisplacement = Math.max(board2.yDisplacement, -1.0f + 0.125f);
		
		//update ball
		ball.xDisplacement += ballXVelocity;
		ball.yDisplacement += ballYVelocity;

		//get next state and update buffer
		Tensor nextState = getState();
		for(int i =0 ; i< STATE_VECTOR_LENGTH; i++) {
			nextStates.set(STATE_VECTOR_LENGTH * experienceCounter + i, nextState.at(i));
		}

		//calculate reward
		float reward = 0.0f;

		//reward based on paddle/ball estimation
		float timeToReachPaddle = (board1.xDisplacement - ball.xDisplacement) / ballXVelocity;
		float predictedY = ball.yDisplacement + ballYVelocity * timeToReachPaddle;

		if (predictedY - 0.025f < -1.0f) predictedY = -2.0f - predictedY;
		if (predictedY + 0.025f >  1.0f) predictedY =  2.0f - predictedY;

		//negative award for being away
		float dist = Math.abs(board1.yDisplacement - predictedY);
		if(dist > 0.13f) {
			reward -= dist;
		}

		
		//check with board 1
		if(ball.xDisplacement - 0.025f < board1.xDisplacement + 0.025f && Math.abs(ball.yDisplacement - board1.yDisplacement) < 0.15f && ball.xDisplacement - 0.025f > board1.xDisplacement - 0.025f) {
				ball.xDisplacement = board1.xDisplacement + 0.05f;
				ballXVelocity *= -1;
				
				//calculate hit pos relative to the center (and adjust the return y velocity based on that)
				float hitPos = (ball.yDisplacement - board1.yDisplacement) / 0.15f;
				ballYVelocity = hitPos * 0.03f; //dummy factor (can change)
				//ball was hit, increase reward
				reward += 0.1f;
				
				//slightly increase ball speed
				ballXVelocity *= 1.01f;
				ballYVelocity *= 1.01f;
		} else if(ball.xDisplacement + 0.025f > board2.xDisplacement - 0.025f && Math.abs(ball.yDisplacement - board2.yDisplacement) < 0.15f && ball.xDisplacement + 0.025f < board2.xDisplacement + 0.025f) {
				ball.xDisplacement = board2.xDisplacement - 0.05f;
				ballXVelocity *= -1;
				
				//calculate hit pos relative to the center (and adjust the return y velocity based on that)
				float hitPos = (ball.yDisplacement - board2.yDisplacement) / 0.15f;
				ballYVelocity = hitPos * 0.03f;
				
				//slightly increase ball speed
				ballXVelocity *= 1.01f;
				ballYVelocity *= 1.01f;
		}
		
		//check for ball collision with top & bottom walls
		if(ball.yDisplacement + 0.025f > 1.0f || ball.yDisplacement - 0.025f < -1.0f) {
				ballYVelocity *= -1;
		}
		
		boolean somebodyLost = false;
		//check for ball leaving play area (somebody gets a point)
		if(ball.xDisplacement + 0.025f < -1f  && frameCounter == 50) {
			//player one lost
			playerTwoPoints++;
			somebodyLost = true;
			//point ball towards who lost
			ballXVelocity = -0.02f;

			//lost the ball, penalty
			reward -= 1;

			//terminal state, mark
			terminals[experienceCounter] = true;

			totPoints++;
		}else if(ball.xDisplacement - 0.025f > 1f && frameCounter == 50) {
			//player two lost
			playerOnePoints++;
			somebodyLost = true;
			//point ball towards who lost
			ballXVelocity = 0.02f;

			//got the point, reward
			reward += 1;

			terminals[experienceCounter] = true;

			nnPoints++;
			totPoints++;
		}
		
		//start the countdown until the ball is reset
		if(somebodyLost || frameCounter < 50) {
			frameCounter--;
		}
		
		//countdown over, reset the ball
		if(frameCounter == 0) {
			frameCounter = 50;
			//reset ball position to random y & random velocity
			ball.yDisplacement = (float) (2 * Math.random() - 1);
			ball.xDisplacement = 0;
			ballYVelocity = (Math.random() > 0.5) ? (0.01f) : (-0.01f);
		}

		//place into buffer
		rewards.set(experienceCounter, reward);

		//train
		if(experienceCounter == datasetSize-1) {
			long diff = System.currentTimeMillis();
			//calculate target q values
			Tensor nextQs = targetNet.predict(nextStates);

			float[] maxNQs = new float[datasetSize];
			for(int i =0; i< datasetSize; i++) {
				float max = -Float.MAX_VALUE;
				for(int j = 0; j< NUM_ACTIONS; j++) {
					max = Math.max(max, nextQs.at(i*NUM_ACTIONS + j));
				}
				maxNQs[i] = max;
			}

			Tensor targetQs = new Tensor(new Shape(datasetSize, NUM_ACTIONS));
			targetQs.init();
			for(int i =0; i< datasetSize; i++) {
				float qup = rewards.at(i);
				if(!terminals[i]) {
					qup += gamma * maxNQs[i];
				}

				for(int j =0; j< NUM_ACTIONS; j++) {
					if(j == actions[i]) {
						targetQs.set(i*NUM_ACTIONS + j, qup);
					}else {
						targetQs.set(i*NUM_ACTIONS + j, predictedQs.at(i*NUM_ACTIONS + j));
					}
				}
			}

			//train
			Tensor[] dataset = {states, targetQs};
			ann.train(dataset, 1, 64, false, optimizer, tracker);
			//decay eps here
			eps *= epsDecay;
			//max it out at 5% of actions are random
			eps = Math.max(eps, 0.05f);

			experienceCounter = 0;
			//reset storages
			states.init();
			predictedQs.init();
			nextStates.init();
			rewards.init();
			actions = new int[datasetSize];
			terminals = new boolean[datasetSize];
			
			diff = System.currentTimeMillis() - diff;
			System.out.println("Epoch " + (networkUpdates+1) + " took " + (((int) diff) / 1000.0f) + " seconds.");
			//print amount of points scored by nn (game metric)
			if(fast) System.out.println("NN scored " + (100 * ((float) nnPoints / (float) totPoints)) + "% of points.");
			nnPoints = 0;
			totPoints = 0;

			networkUpdates++;
			if(networkUpdates % 10 == 0) {
				//update target network
				ann.save("targetNet_" + networkUpdates + ".bin");
				targetNet = Ann.load("targetNet_"+ networkUpdates + ".bin");
			}

			if(networkUpdates % 100 == 0) {
				fast = false;
			}else {
				fast = true;
			}
		}
		
		if(playerOnePoints > 10 || playerTwoPoints > 10) {
			//game over
			numGames++;
			resetGame();
		}

		experienceCounter++;
	}
	
	public void resetGame() {
		board1.yDisplacement = 0;
		board2.yDisplacement = 0;
		ball.yDisplacement = 0;
		ball.xDisplacement = 0;
		ballXVelocity = -0.01f;
		ballYVelocity = 0;
		board1Velocity = 0.0f;
		board2Velocity = 0.0f;
		frameCounter = 50;
		if(!fast) System.out.println("Game " + numGames + ": Player 1: "+ playerOnePoints + ", Player 2: " + playerTwoPoints);
		playerOnePoints = 0;
		playerTwoPoints = 0;
	}
	
	//-------------ANN stuff ---------------------------
	
	public int getAnnAction(Tensor state) {
		//forward prop through ann
		Tensor expected = ann.predict(state);
		//plop into buffer
		for(int i = 0; i< NUM_ACTIONS; i++) {
			predictedQs.set(experienceCounter * NUM_ACTIONS + i, expected.at(i));
		}

		//epsilon-greedy selection
		double rand = Math.random();
		if(rand < eps) {
			if(rand < 0.333333f) {
				return 0;
			}
			if(rand < 0.66667f) {
				return 1;
			}
			return -1;
		}else {
			int max = 0;
			for(int i = 1; i< NUM_ACTIONS; i++) {
				if(expected.at(i) > expected.at(max)) {
					max = i;
				}
			}
			return max;
		}
	}
	
	public Tensor getInitialState() {
		float[] state = {
			0.0f, 0.0f, //initial ball displacement	
			-0.01f, 0.0f, //initial ball velocity
			
			0.0f, 0.0f,//initial paddle 1 y displacement + velocity
			0.0f, 0.0f // initial paddle 2 y displacement + velocity
		};
		Tensor t = new Tensor(new Shape (STATE_VECTOR_LENGTH));
		t.init(state);
		return t;
	}
	
	public Tensor getState() {
		float[] state = {
			ball.xDisplacement, ball.yDisplacement, //ball displacement	
			ballXVelocity, ballYVelocity, //ball velocity
			
			board1.yDisplacement, board1Velocity, //paddle 1 y displacement + velocity
			board2.yDisplacement, board2Velocity // paddle 2 y displacement + velocity
		};
		Tensor t = new Tensor(new Shape (STATE_VECTOR_LENGTH));
		t.init(state);
		return t;
	}
	
	
}