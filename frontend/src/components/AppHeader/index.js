
import { Space, Typography } from "antd";
import { useAuth } from '../../common/AuthProvider'
import {
  UserOutlined,
} from "@ant-design/icons";
function AppHeader() {
  const { user, handleLogout } = useAuth()

  const logOut = () => {
    handleLogout()
  
  }
  return (
    <div className="AppHeader">
      {/* Left Side: Title */}
      <Typography.Title level={4} style={{ margin: 0 }}>Metadata-Filter RAG</Typography.Title>
        
      <Space align="center" style={{ marginLeft: 'auto' }}>
        

        {/* User Icon */}
        <UserOutlined style={{ fontSize: '1.5em' }} />

        {/* Display Current User */}
        <Typography.Text strong className="ml-2">
          {user.sub} - Role: {user.role} - Department: {user.department}
        </Typography.Text>

        {/* User Role */}

        {/* Logout Button */}
        <button
          type="text"
          danger
          onClick={logOut}
          className="px-2 text-white bg-red-600 rounded-md hover:bg-red-700 active:shadow-lg"
        >
          Log Out
        </button>
      </Space>
    </div>
  );
}
export default AppHeader;
