package util;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.CatanMlp;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.HashMap;
import java.util.Map;

/**
 * Project utility class to save and load models and parameters.
 * 
 */
public class ModelUtils {

    private static final Logger log = LoggerFactory.getLogger(ModelUtils.class);
    public static final String PARAMS_PATH = "./models/";
    public static final String AUTOENC_SUBPATH = "autoencoder/";
    public static final String MT_SUBPATH = "mt/";
    public static final String MLP_SUBPATH = "mlp/";
    public static final String FEATURE_MASK = "mask/";
    
    
    private ModelUtils(){}
  
    /**
     * Method to save a catan mlp network and it's configuration
     * @param net
     * @param basePath
     */
    public static void saveMlpModelAndParameters(CatanMlp net, int task) {
        String confPath = FilenameUtils.concat(PARAMS_PATH + MLP_SUBPATH, MultiLayerConfiguration.class.getName() + "-" + task + "-conf.json");
        String paramPath = FilenameUtils.concat(PARAMS_PATH + MLP_SUBPATH, net.getClass().getName() + "-" + task + ".bin");
        log.info("Saving model and parameters to {} and {} ...",  confPath, paramPath);

        //make sure the folder structure exists
        File f = new File(paramPath);
        f.getParentFile().mkdirs();
        f = new File(confPath);
        f.getParentFile().mkdirs();
        
        // save parameters
        try {
            DataOutputStream dos = new DataOutputStream(new FileOutputStream(paramPath));
            Nd4j.write(net.params(), dos);
            dos.flush();
            dos.close();

            // save model configuration
            FileUtils.write(new File(confPath), net.getFullConfig().toJson());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
    /**
     * Method to load a catan mlp network
     * TODO: configuration loading needs to be fixed
     * @param net
     * @param basePath
     */
    public static CatanMlp loadMlpModelAndParameters(int task, MultiLayerConfiguration conf) {
        log.info("Loading saved model and parameters...");
        CatanMlp savedNetwork = null;
        String paramPath = PARAMS_PATH + MLP_SUBPATH + CatanMlp.class.getName() + "-" + task + ".bin";
//        String confPath = PARAMS_PATH + MLP_SUBPATH + MultiLayerConfiguration.class.getName() + "-" + task + "-conf.json";
        // load parameters
        try {
//            MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(FileUtils.readFileToString(new File(confPath)));
            DataInputStream dis = new DataInputStream(new FileInputStream(paramPath));
            INDArray newParams = Nd4j.read(dis);
            dis.close();
            // load model configuration
            savedNetwork = new CatanMlp(conf);
            savedNetwork.init();
            savedNetwork.setParams(newParams);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return savedNetwork;
    }
    
/////Methods for save/load specific layers///////    
    
    public static void saveLayerParameters(INDArray param, String paramPath)  {
        // save parameters for each layer
        log.info("Saving parameters to {} ...", paramPath);

        try {
            DataOutputStream dos = new DataOutputStream(new FileOutputStream(paramPath));
            Nd4j.write(param, dos);
            dos.flush();
            dos.close();
        } catch(IOException e) {
            e.printStackTrace();
        }
    }

    public static Layer loadLayerParameters(Layer layer, String paramPath) {
        // load parameters for each layer
//        String name = layer.conf().getLayer().getLayerName();
        log.info("Loading saved parameters for layer {} ...");

        try{
        DataInputStream dis = new DataInputStream(new FileInputStream(paramPath));
        INDArray param = Nd4j.read(dis);
        dis.close();
        layer.setParams(param);
        } catch(IOException e) {
            e.printStackTrace();
        }

        return layer;
    }

    public static void saveParameters(MultiLayerNetwork model, int[] layerIds, Map<Integer, String> paramPaths) {
        Layer layer;
        for(int layerId: layerIds) {
            layer = model.getLayer(layerId);
            if (!layer.paramTable().isEmpty()) {
                ModelUtils.saveLayerParameters(layer.params(), paramPaths.get(layerId));
            }
        }
    }

    public static MultiLayerNetwork loadParameters(MultiLayerNetwork model, int[] layerIds, Map<Integer, String> paramPaths) {
        Layer layer;
        for(int layerId: layerIds) {
            layer = model.getLayer(layerId);
            loadLayerParameters(layer, paramPaths.get(layerId));
        }
        return model;
    }


    public static  Map<Integer, String>  getIdParamPaths(MultiLayerNetwork model, String basePath, int[] layerIds){
        Map<Integer, String> paramPaths = new HashMap<>();
        for (int id : layerIds) {
            paramPaths.put(id, FilenameUtils.concat(basePath, id + ".bin"));
        }

        return paramPaths;
    }

    public static Map<String, String> getStringParamPaths(MultiLayerNetwork model, String basePath, String[] layerIds){
        Map<String, String> paramPaths = new HashMap<>();

        for (String name : layerIds) {
            paramPaths.put(name, FilenameUtils.concat(basePath, name + ".bin"));
        }

        return paramPaths;
    }
    
}