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
	public static int MAX_EPISODES = 1_000_000;
	public static int datasetSize = 1_000;
	public static int STATE_VECTOR_LENGTH = 8;
	public static int NUM_ACTIONS = 3;

	Tensor states;
	int[] actions;

	Ann ann;

	int epochs = 0;
	int experienceCounter = 0;

	int nnPoints;
	int totPoints;

	float eps = 1.0f;
	float epsDecay = 0.99f;
	float gamma = 0.95f;

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
		
		resetGame();
		
		//init anns
		Input input = new Input(new Shape(STATE_VECTOR_LENGTH));

		Dense hidden = new Dense(input, new Shape(STATE_VECTOR_LENGTH), true);
		
		Dense lay = new Dense(hidden, new Shape(NUM_ACTIONS), false);
		Activation output = new Activation(lay, ActivationFunction.SOFTMAX);
		ann = new Ann(input, output);
		
		ann.printSummary();

		//initialize optimizer and tracker
		//lr = 3e-4
		optimizer = new AdamOptimizer(1f, 0.9f, 0.999f);
		//track loss, don't track accuracy, and print it
		tracker = new ann.Metrics(false, false, true);

		//initialize storage
		states = new Tensor(new Shape(datasetSize, STATE_VECTOR_LENGTH));
		states.init();

		actions = new int[datasetSize];

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

		ann.save("final");
	}

	public void update() {
		//get current state from game
		Tensor state = getState();
		//propagate through ann and select from the probability distribution outputted
		int action = getAnnAction(state);
		//get idx (explained in the below comment)
		int idx = experienceCounter % datasetSize;
		//update buffers
		for(int i= 0; i< STATE_VECTOR_LENGTH; i++) {
			//here we overwrite the oldest entries by modding the index by datasetSize
			//then, in training time, we can rewind from here for discounting
			states.set(STATE_VECTOR_LENGTH * idx + i, state.at(i));
		}
		//similarly overwrite oldest action here
		actions[idx] = action;

		//now propagate action to forward movement
		if(action == 0) {
			board1Velocity = 0.05f;
		}else if(action == 1) {
			board1Velocity = -0.05f;
		}else{
			board1Velocity = 0.0f;
		}
		board1.yDisplacement += board1Velocity;
	    
	  //simple board2 policy
	  float displacement = 0.075f * (ball.yDisplacement - board2.yDisplacement);
	  board2Velocity = Math.signum(displacement) * Math.min(Math.abs(displacement), 0.03f);
	  board2.yDisplacement += board2Velocity;
	    
	    
		//propagating ann action to next state
		board1.yDisplacement = Math.min(board1.yDisplacement, 1.0f - 0.125f);
		board1.yDisplacement = Math.max(board1.yDisplacement, -1.0f + 0.125f);
		
		board2.yDisplacement = Math.min(board2.yDisplacement, 1.0f - 0.125f);
		board2.yDisplacement = Math.max(board2.yDisplacement, -1.0f + 0.125f);
		
		//update ball
		ball.xDisplacement += ballXVelocity;
		ball.yDisplacement += ballYVelocity;

		//check for collision with board 1
		if(ball.xDisplacement - 0.025f < board1.xDisplacement + 0.025f && Math.abs(ball.yDisplacement - board1.yDisplacement) < 0.15f && ball.xDisplacement - 0.025f > board1.xDisplacement - 0.025f) {
				//set to border and flip velocity
				ball.xDisplacement = board1.xDisplacement + 0.05f;
				ballXVelocity *= -1;
				
				//calculate hit pos relative to the center (and adjust the return y velocity based on that)
				float hitPos = (ball.yDisplacement - board1.yDisplacement) / 0.15f;
				ballYVelocity = hitPos * 0.03f; //dummy factor (can change)
				
				//slightly increase ball speed
				ballXVelocity *= 1.01f;
				ballYVelocity *= 1.01f;
				
				//board 2 next
		} else if(ball.xDisplacement + 0.025f > board2.xDisplacement - 0.025f && Math.abs(ball.yDisplacement - board2.yDisplacement) < 0.15f && ball.xDisplacement + 0.025f < board2.xDisplacement + 0.025f) {
				//set to paddle border and flip velocity
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
				//flip velocity
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

			totPoints++;
		}else if(ball.xDisplacement - 0.025f > 1f && frameCounter == 50) {
			//player two lost
			playerOnePoints++;
			somebodyLost = true;
			//point ball towards who lost
			ballXVelocity = 0.02f;

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
			ballXVelocity = Math.signum((float) Math.random() - 0.05f) * 0.02f;

			//set ys to 0
			board1.yDisplacement = 0.0f;
			board2.yDisplacement = 0.0f;
		}
		
		if(playerOnePoints > 10 || playerTwoPoints > 10) {
			//game over
			numGames++;
			resetGame();

			//create custom scale tensor
			//this is gamma discounting each example
			//since we want further away actions to be weighted less, we will go from idx and work backwards (loop around 0 if necessary)

			//use broadcasting, so only need one scalar per example (datasetSize examples)
			Tensor customScale = new Tensor(new Shape(1, datasetSize));
			customScale.init();

			for(int i = 0; i < datasetSize; i++) {
				int mapped = (idx - i + datasetSize) % datasetSize;
				float discount = (float) (Math.pow(gamma, i));
				customScale.set(mapped, discount);
			}

			//update optimizer object
			optimizer.customScale(customScale);
			//positive=true means reinforce (make probability of this happening higher), positive=false means discourage (make probability of this happening lower)
			//however gradient direction is opposite (descent = make prob higher, ascent = make prob lower), so switch sign
			optimizer.setGradientDirection(playerTwoPoints == 11);

			//train!!
			Tensor[] dataset = {states, oneHot(actions, NUM_ACTIONS)};
			ann.train(dataset, 1, 50, false, optimizer, tracker);

			//decay epsilon
			if(epochs % 50 == 0) eps *= epsDecay;
			//minimum 5% of actions are random
			eps = Math.max(eps, (epochs < 10_000) ? 0.05f : 0.0f);

			//set experience counter back to 0 and reset storages
			experienceCounter = 0;

			states.init();
			actions = new int[datasetSize];

			//epoch done, increment counter
			epochs++;

			nnPoints = 0;
			totPoints = 0;
			
			
			//save if % 10
			if(epochs % 10000 == 0) {
				ann.save("targetNet_" + (epochs/50000) + ".bin");
			}

			fast = !(epochs % 10000 == 0);
		}

		experienceCounter++;
	}
	
	public void resetGame() {
		board1.yDisplacement = 0;
		board2.yDisplacement = 0;
		ball.xDisplacement = 0;
		//reset ball position to random y & random velocity
		ball.yDisplacement = (float) (2 * Math.random() - 1);
		ball.xDisplacement = 0;
		ballYVelocity = (Math.random() > 0.5) ? (0.01f) : (-0.01f);
		ballXVelocity = Math.signum((float) Math.random() - 0.05f) * 0.02f;
		board1Velocity = 0.0f;
		board2Velocity = 0.0f;
		frameCounter = 50;
		if(fast && epochs % 1000 == 0) {
			System.out.println("Game " + numGames + ": Player 1: "+ playerOnePoints + ", Player 2: " + playerTwoPoints + "\t Randomness: " + 100.0f * eps);
		}
		playerOnePoints = 0;
		playerTwoPoints = 0;
	}
	
	//-------------ANN stuff ---------------------------
	
	public int getAnnAction(Tensor state) {
	// Forward pass
	Tensor probs = ann.predict(state); //shape = (NUM_ACTIONS)

	//epsilon greedy, explore based on epsilon value
	if (Math.random() < eps) {
		return (int)(Math.random() * NUM_ACTIONS);//choose a random action
	}

	//sample from the probability distribution outputted by the network
	double r = Math.random();
	double cumulative = 0.0;
	for (int i = 0; i < NUM_ACTIONS; i++) {
		cumulative += probs.at(i);
		if (r < cumulative) {
			return i; //select action
		}
	}

	//choose standing still by def
	return NUM_ACTIONS - 1;
}


	public Tensor oneHot(int action, int numActions) {
		Tensor res = new Tensor(new Shape(numActions));
		res.init();
		res.set(action, 1.0f);
		return res;
	}

	//up but batched!!
	public Tensor oneHot(int[] action, int numActions) {
		Tensor res = new Tensor(new Shape(action.length, numActions));
		res.init();
		for(int i =0 ; i< action.length; i++) {
			res.set(i * numActions + action[i], 1.0f);
		}

		return res;
	}
	
	public Tensor getState() {
		float[] state = {
			ball.xDisplacement, ball.yDisplacement, //ball displacement	
			ballXVelocity, ballYVelocity, //ball velocity
			
			board1.yDisplacement, board1Velocity, //paddle 1 y displacement
			board2.yDisplacement, board2Velocity // paddle 2 y displacement
		};
		Tensor t = new Tensor(new Shape (STATE_VECTOR_LENGTH));
		t.init(state);
		return t;
	}
	
	
}