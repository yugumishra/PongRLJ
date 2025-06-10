package environment;

import java.util.Random;

import org.lwjgl.glfw.GLFW;

import ann.Activation;
import ann.ActivationFunction;
import ann.Ann;
import ann.Dense;
import ann.Dropout;
import ann.Input;
import ann.Shape;
import ann.Tensor;

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
	
	int frames;
	
	int frameCounter = 50;
	int numGames;
	
	// ---------- ANN variables ---------------
	public static int MAX_EPISODES = 100_000;
	public static int datasetSize = 10_000;
	public static int STATE_VECTOR_LENGTH = 8;

	Tensor states;
	int[] actions;
	Tensor rewards;
	Tensor nextStates;
	boolean[] terminals;

	Ann ann;
	Ann targetNet;

	int networkUpdates = 0;
	
	public Pong() {
		frames = 0;
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
		
		Dense hidden = new Dense(input, new Shape(16), true);
		Activation activation = new Activation(hidden, ActivationFunction.RELU);
		Dropout drop = new Dropout(activation, 0.5f);
		
		Dense fin = new Dense(drop, new Shape(3), true);
		Activation output = new Activation(fin, ActivationFunction.SOFTMAX);
		
		ann = new Ann(input, output);
		
		ann.printSummary();

		//save network into target file, then read it for the target network
		ann.save("targetNet_" + networkUpdates + ".bin");

		targetNet = Ann.load("targetNet_"+ networkUpdates + ".bin");

		//initialize storage
		states = new Tensor(new Shape(datasetSize, STATE_VECTOR_LENGTH));
		nextStates = new Tensor(new Shape(datasetSize, STATE_VECTOR_LENGTH));
		states.init();
		nextStates.init();

		rewards = new Tensor(new Shape(datasetSize));
		rewards.init();

		actions = new int[datasetSize];
		terminals = new boolean[datasetSize];
	}
	
	public void play() {
		//loop
		while(true) {
			w.clear();
			w.draw(board1);
			w.draw(board2);
			if(frameCounter == 50) w.draw(ball);
			for(Drawable d: divider) w.draw(d);
			w.update();
			
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
		
		//check with board 1
		if(ball.xDisplacement - 0.025f < board1.xDisplacement + 0.025f && Math.abs(ball.yDisplacement - board1.yDisplacement) < 0.15f && ball.xDisplacement - 0.025f > board1.xDisplacement - 0.025f) {
				ball.xDisplacement = board1.xDisplacement + 0.05f;
				ballXVelocity *= -1;
				
				//calculate hit pos relative to the center (and adjust the return y velocity based on that)
				float hitPos = (ball.yDisplacement - board1.yDisplacement) / 0.15f;
				ballYVelocity = hitPos * 0.03f; //dummy factor (can change)
				
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
		}else if(ball.xDisplacement - 0.025f > 1f && frameCounter == 50) {
			//player two lost
			playerOnePoints++;
			somebodyLost = true;
			//point ball towards who lost
			ballXVelocity = 0.02f;
		}
		
		if(somebodyLost || frameCounter < 50) {
			frameCounter --;
		}
		
		if(frameCounter == 0) {
			frameCounter = 50;
			//reset ball position to random y & random velocity
			ball.yDisplacement = (float) (2 * Math.random() - 1);
			ball.xDisplacement = 0;
			ballYVelocity = (Math.random() > 0.5) ? (0.01f) : (-0.01f);
		}
		
		if(playerOnePoints > 10 || playerTwoPoints > 10) {
			//game over
		numGames++;
			resetGame();
			
			/*
			if(numGames % batchSize == 0) {
				//train per batch
				for(int i = 0; i< rollout.length; i++) {
					Experience e = rollout[i];
					
				}
				
				
				
				//eps decay (so ann can get better (with less random actions))
			if(numGames % 5 == 0) {
				eps *= 0.99;
			}
			}*/
		
		
		}
		
		frames++;
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
		System.out.println("Game " + numGames + ": Player 1: "+ playerOnePoints + ", Player 2: " + playerTwoPoints);
		playerOnePoints = 0;
		playerTwoPoints = 0;
	}
	
	//-------------ANN stuff ---------------------------
	
	public int getAnnAction(Tensor state) {
		/*
		double rand = Math.random();
		if(rand < eps) {
			if(rand - eps/2 > 0) {
				return 0;
			}else {
				return 1;
			}
		}*/
		//forward prop through ann
		Tensor expected = ann.predict(state);
		//final layer is 2 neurons, one for the probability of each action
		//sample from said probability distribution
		Random random = new Random();
		float r = random.nextFloat();
		
		float cum = 0.0f;
		int action = -1;
		
		for(int i = 0; i< 3; i++) {
			cum += expected.at(i);
			if(r < cum) {
				action = i;
				if(i == 3) action = -1;
				break;
			}
		}
		
		return action;
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