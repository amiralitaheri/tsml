package utilities;

import weka.core.OptionHandler;

import java.util.Enumeration;

public interface IndividualOptionHandler
    extends OptionHandler {
    @Override
    default Enumeration listOptions() {
        throw new UnsupportedOperationException();
    }

    @Override
    default void setOptions(final String[] options) throws
                                                   Exception {
        if(options.length % 2 != 0) {
            throw new IllegalArgumentException("options must be array of even length of key pair values");
        }
        for(int i = 0; i < options.length; i += 2) {
            setOption(options[i], options[i + 1]);
        }
    }

    void setOption(String key, String value);

    @Override
    String[] getOptions();
}